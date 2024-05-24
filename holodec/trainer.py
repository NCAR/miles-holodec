import gc
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def cleanup():
    dist.destroy_process_group()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


class Trainer:

    def __init__(self, model, rank, module=False):
        super(Trainer, self).__init__()
        self.model = model
        self.rank = rank
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

        if module:
            self.model = self.model.module

    # Training function.
    def train_one_epoch(
        self,
        epoch,
        conf,
        trainloader,
        optimizer,
        criterion,
        scaler,
        scheduler,
        metrics
    ):

        batches_per_epoch = conf['trainer']['batches_per_epoch']
        grad_accum_every = conf['trainer']['grad_accum_every']
        amp = conf['trainer']['amp']
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "lambda":
            scheduler.step()

        batches_per_epoch = (
            batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)
        )

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        self.model.train()

        dl = cycle(trainloader)

        for steps in range(batches_per_epoch):

            logs = {}

            for _ in range(grad_accum_every):

                x, y_part_mask, y_depth_mask, y_weight_mask = next(dl)

                with autocast(enabled=amp):

                    y_pred = self.model(x)
                    y_pred_mask = y_pred[:, 0]
                    y_pred_depth = y_pred[:, 1]

                    y_part_mask = y_part_mask.to(y_pred.device, y_pred.dtype)
                    y_depth_mask = y_depth_mask.to(y_pred.device, y_pred.dtype)
                    y_weight_mask = y_weight_mask.to(y_pred.device, y_pred.dtype)

                    mask_loss = criterion[0](y_part_mask, y_pred_mask)#, y_weight_mask)
                    depth_loss = criterion[1](y_depth_mask*y_part_mask, y_pred_depth*y_part_mask)

                    # Mask metrics
                    for name, metric in metrics[0].items():
                        value = metric(y_part_mask, y_pred_mask)
                        value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                        if distributed:
                            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                        results_dict[f"train_{name}"].append(value[0].item())

                    # Depth metrics
                    for name, metric in metrics[1].items():
                        value = metric(y_depth_mask*y_part_mask, y_pred_depth*y_part_mask)
                        value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                        if distributed:
                            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                        results_dict[f"train_{name}"].append(value[0].item())

                    loss = mask_loss + conf['loss'].get('training_loss_depth_mult',1.0)*depth_loss

                    scaler.scale(loss / grad_accum_every).backward()

                accum_log(logs, {'loss': loss.item() / grad_accum_every})
                accum_log(logs, {'mask_loss': mask_loss.mean().item() / grad_accum_every})
                accum_log(logs, {'depth_loss': depth_loss.mean().item() / grad_accum_every})

            # backward

            if distributed:
                torch.distributed.barrier()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            for key in logs:
                batch_loss = torch.Tensor([logs[key]]).cuda(self.device)
                if distributed:
                    dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                results_dict[f"train_{key}"].append(batch_loss[0].item())

            # update the learning rate if epoch-by-epoch updates that dont depend on a metric
            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "cosine-annealing":
                scheduler.step()

            # agg the results
            to_print = "Epoch (train): {} loss: {:.6f} mask_loss: {:.6f} depth_loss: {:.6f}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_mask_loss"]),
                np.mean(results_dict["train_depth_loss"]),
                #np.mean(results_dict["train_acc"]),
                #np.mean(results_dict["train_mae"])
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(
        self,
        epoch,
        conf,
        valid_loader,
        criterion,
        metrics
    ):

        self.model.eval()

        valid_batches_per_epoch = conf['trainer']['valid_batches_per_epoch']
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        valid_batches_per_epoch = (
            valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)
        )

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )

        with torch.no_grad():
            for k, (x, y_part_mask, y_depth_mask, y_weight_mask) in enumerate(valid_loader):

                y_pred = self.model(x)

                y_part_mask = y_part_mask.to(y_pred.device, y_pred.dtype)
                y_depth_mask = y_depth_mask.to(y_pred.device, y_pred.dtype)
                y_weight_mask = y_weight_mask.to(y_pred.device, y_pred.dtype)

                mask_loss = criterion[0](y_part_mask, y_pred[:, 0])
                depth_loss = criterion[1](y_depth_mask*y_part_mask, y_pred[:, 1]*y_part_mask)

                # Mask metrics
                for name, metric in metrics[0].items():
                    value = metric(y_part_mask, y_pred[:, 0])
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"valid_{name}"].append(value[0].item())

                # Depth metrics
                for name, metric in metrics[1].items():
                    value = metric(y_depth_mask, y_pred[:, 1])
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"valid_{name}"].append(value[0].item())

                loss = (mask_loss + conf['loss'].get('validation_loss_depth_mult',1.0)*depth_loss).mean()

                for key, _loss in zip(["loss", "mask_loss", "depth_loss"], [loss, mask_loss, depth_loss]):
                    batch_loss = torch.Tensor([_loss.mean().item()]).cuda(self.device)
                    if distributed:
                        torch.distributed.barrier()
                    results_dict[f"valid_{key}"].append(batch_loss[0].item())

                # agg the results
                to_print = "Epoch (valid): {} loss: {:.6f} mask_loss: {:.6f} depth_loss: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_mask_loss"]),
                    np.mean(results_dict["valid_depth_loss"]),
                )

                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

                if k >= valid_batches_per_epoch and k > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def fit(
        self,
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        trial=False
    ):
        save_loc = conf['save_loc']
        start_epoch = conf['trainer']['start_epoch']
        epochs = conf['trainer']['epochs']

        # Reload the results saved in the training csv if continuing to train
        if start_epoch == 0:
            results_dict = defaultdict(list)
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(f"{save_loc}/training_log.csv")
            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        for epoch in range(start_epoch, epochs):

            ############
            #
            # Train
            #
            ############

            train_results = self.train_one_epoch(
                epoch,
                conf,
                train_loader,
                optimizer,
                train_criterion,
                scaler,
                scheduler,
                metrics
            )

            ############
            #
            # Checkpoint
            #
            ############

            if not trial:

                if conf["trainer"]["mode"] != "fsdp":

                    # Save the current model

                    logging.info(f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

                    state_dict = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),  #self.model.module.state_dict() if conf["trainer"]["mode"] == "ddp" else self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict()
                    }

                    #torch.save(state_dict, f"{save_loc}/checkpoint_{self.device}.pt" if conf["trainer"]["mode"] == "ddp" else f"{save_loc}/checkpoint.pt")
                    torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                else:

                    logging.info(f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

                    # https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
                    FSDP.set_state_dict_type(
                        self.model,
                        StateDictType.SHARDED_STATE_DICT,
                    )
                    sharded_state_dict = {
                        "model_state_dict": self.model.state_dict()
                    }
                    DCP.save_state_dict(
                        state_dict=sharded_state_dict,
                        storage_writer=DCP.FileSystemWriter(os.path.join(save_loc, "checkpoint")),
                    )
                    # save the optimizer
                    optimizer_state = FSDP.full_optim_state_dict(self.model, optimizer)
                    state_dict = {
                        "epoch": epoch,
                        "optimizer_state_dict": optimizer_state,
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict()
                    }

                    torch.save(state_dict, f"{save_loc}/checkpoint.pt")

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()

            ############
            #
            # Validation
            #
            ############

            valid_results = self.validate(
                epoch,
                conf,
                valid_loader,
                valid_criterion,
                metrics
            )

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()

            #################
            #
            # Save and Return
            #
            #################

            # Put things into a results dictionary -> dataframe
            ###
            ### This needs updated!
            ###
            ###
            results_dict["epoch"].append(epoch)
            for name in ["loss", "mae"]:
                results_dict[f"train_{name}"].append(np.mean(train_results[f"train_{name}"]))
                results_dict[f"valid_{name}"].append(np.mean(valid_results[f"valid_{name}"]))
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            df = pd.DataFrame.from_dict(results_dict).reset_index()

            # update the learning rate if epoch-by-epoch updates

            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "plateau":
                scheduler.step(results_dict["valid_mae"][-1])

#             # Save the best model so far
#             if not trial:

#                 # This needs updated!
#                 valid_loss = np.mean(valid_results["valid_loss"])

#                 # save if this is the best model seen so far
#                 if (self.device == 'cuda:0') and (np.mean(valid_loss) == min(results_dict["valid_loss"])):
#                     if conf["trainer"]["mode"] == "ddp":
#                         shutil.copy(f"{save_loc}/checkpoint_{self.device}.pt", f"{save_loc}/best_{self.device}.pt")
#                     elif conf["trainer"]["mode"] == "fsdp":
#                         if os.path.exists(f"{save_loc}/best"):
#                             shutil.rmtree(f"{save_loc}/best")
#                         shutil.copytree(f"{save_loc}/checkpoint", f"{save_loc}/best")
#                     else:
#                         shutil.copy(f"{save_loc}/checkpoint.pt", f"{save_loc}/best.pt")

            # Save the dataframe to disk
            if trial:
                df.to_csv(
                    f"{save_loc}/trial_results/training_log_{trial.number}.csv",
                    index=False,
                )
            else:
                df.to_csv(f"{save_loc}/training_log.csv", index=False)

            # Report result to the trial
            #if trial:
                # trial.report
                #pass

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict["valid_loss"])
                if j == min(results_dict["valid_loss"])
            ][0]
            offset = epoch - best_epoch
            if offset >= conf['trainer']['stopping_patience']:
                logging.info(f"Trial {trial.number} is stopping early")
                break

        best_epoch = [
            i for i, j in enumerate(results_dict["valid_loss"]) if j == min(results_dict["valid_loss"])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
