from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import shutil
import torch
import yaml
import tqdm
import sys
import gc

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from holodec.datasets import UpsamplingReader, XarrayReader
# from holodec.transforms import LoadTransformations
from holodec.planer_transforms import LoadTransformations
from holodec.pbs import launch_script, launch_script_mpi
from holodec.unet import PlanerSegmentationModel as SegmentationModel
from holodec.losses import load_loss
from holodec.seed import seed_everything
from holodec.scheduler import load_scheduler

import os
import warnings
warnings.filterwarnings("ignore")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, conf, mode="ddp", trial=False):
    # Set up dist
    setup(rank, world_size, mode)

    # Set seeds for reproducibility
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    epochs = conf["trainer"]["epochs"]
    start_epoch = 0 if "start_epoch" not in conf["trainer"] else conf["trainer"]["start_epoch"]
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    model_loc = conf["save_loc"]

    training_loss = "dice-bce" if "training_loss_mask" not in conf["loss"] else conf["loss"]["training_loss_mask"]
    valid_loss = "dice" if "validation_loss_mask" not in conf[
        "loss"] else conf["loss"]["validation_loss_mask"]

    # Set up CUDA/CPU devices
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

    # Load the preprocessing transforms
    if "Normalize" in conf["transforms"]["training"]:
        conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]
        conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]

    train_transforms = LoadTransformations(conf["transforms"]["training"])
    valid_transforms = LoadTransformations(conf["transforms"]["validation"])

    # Load the datasets
    train_dataset = XarrayReader(
        conf["train_dataset"]["path"],
        train_transforms,
        mode="mask"
    )
    test_dataset = XarrayReader(
        conf["validation_dataset"]["path"],
        valid_transforms,
        mode="mask"
    )

    # Load distributed samplers
    sampler_tr = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # May be True
        seed=seed,
        drop_last=True
    )

    sampler_val = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=False,
        drop_last=True
    )

    # Load the iterators for batching the data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_tr,
        batch_size=train_batch_size,
        num_workers=conf["trainer"]["thread_workers"],
        pin_memory=False,
        shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_val,
        batch_size=valid_batch_size,
        num_workers=conf["trainer"]["valid_thread_workers"],
        pin_memory=False,
        shuffle=False)

    # Load a segmentation model
    unet = SegmentationModel(conf).to(device)
    unet = DDP(unet, device_ids=[f"cuda:{rank}"], output_device=None)

    if start_epoch > 0:
        # Load weights
        logging.info(f"Restarting training starting from epoch {start_epoch}")
        logging.info(f"Loading model weights from {model_loc}")
        checkpoint = torch.load(
            os.path.join(model_loc, "best.pt"),
            map_location=lambda storage, loc: storage
        )
        unet.load_state_dict(checkpoint["model_state_dict"])

    unet = unet.to(device)

    # Load an optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), **conf["optimizer"])

    if start_epoch > 0:
        # Load weights
        logging.info(f"Loading optimizer state from {model_loc}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Specify the training and validation losses
    train_criterion = load_loss(training_loss)
    test_criterion = load_loss(valid_loss, split="validation")

    # Load a learning rate scheduler
    lr_scheduler = load_scheduler(optimizer, conf)
    # lr_scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     patience=1,
    #     min_lr=1.0e-13,
    #     verbose=True
    # )

    # Reload the results saved in the training csv if continuing to train
    if start_epoch == 0:
        results_dict = defaultdict(list)
        epoch_test_losses = []
    else:
        results_dict = defaultdict(list)
        saved_results = pd.read_csv(f"{model_loc}/training_log.csv")
        epoch_test_losses = list(saved_results["valid_loss"])
        for key in saved_results.columns:
            if key == "index":
                continue
            results_dict[key] = list(saved_results[key])
        # update the learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Train a U-net model
    for epoch in range(start_epoch, epochs):

        # Train the model
        unet.train()

        batch_loss = []

        # set up a custom tqdm
        batch_group_generator = tqdm.tqdm(
            enumerate(train_loader),
            total=batches_per_epoch,
            leave=True
        )

        for k, (inputs, y) in batch_group_generator:
            # Move data to the GPU, if not there already
            inputs = inputs.to(device)
            y = y.to(device)

            # Clear gradient
            optimizer.zero_grad()

            # get output from the model, given the inputs
            pred_mask = unet(inputs)

            # get loss for the predicted output
            loss = train_criterion(pred_mask, y.float())

            # get gradients w.r.t to parameters
            loss.backward()

            # Average across all GPUs and report
            dist.all_reduce(loss, dist.ReduceOp.AVG, async_op=False)
            batch_loss.append(loss.item())

            # update parameters
            optimizer.step()

            # update tqdm
            to_print = "Epoch {} train_loss: {:.4f}".format(
                epoch, np.mean(batch_loss))
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]['lr'])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            # stop the training epoch when train_batches_per_epoch have been used to update
            # the weights to the model
            if k >= batches_per_epoch and k > 0:
                break

            if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "cosine-annealing":
                lr_scheduler.step()

        # Shutdown the progbar
        batch_group_generator.close()

        # Compuate final performance metrics before doing validation
        train_loss = np.mean(batch_loss)

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # Test the model
        unet.eval()
        with torch.no_grad():

            batch_loss = []

            # set up a custom tqdm
            batch_group_generator = tqdm.tqdm(
                enumerate(test_loader),
                total=valid_batches_per_epoch,
                leave=True
            )

            for k, (inputs, y) in batch_group_generator:
                # Move data to the GPU, if not there already
                inputs = inputs.to(device)
                y = y.to(device)
                # get output from the model, given the inputs
                pred_mask = unet(inputs)
                # get loss for the predicted output
                loss = test_criterion(pred_mask, y.float())
                batch_loss.append(loss.item())
                # update tqdm
                to_print = "Epoch {} test_loss: {:.4f}".format(
                    epoch, np.mean(batch_loss))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

            # Shutdown the progbar
            batch_group_generator.close()

        # Use the supplied metric in the config file as the performance metric to toggle learning rate and early stopping
        test_loss = np.mean(batch_loss)
        epoch_test_losses.append(test_loss)

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # Lower the learning rate if we are not improving
        if conf['trainer']['use_scheduler'] and conf['trainer']['scheduler']['scheduler_type'] == "plateau":
            lr_scheduler.step(test_loss)

        # Save the model if its the best so far.
        if test_loss == min(epoch_test_losses):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict() if mode != "ddp" else unet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                'loss': test_loss
            }
            torch.save(state_dict, f"{model_loc}/best.pt")

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]['lr']

        # Put things into a results dictionary -> dataframe
        results_dict['epoch'].append(epoch)
        results_dict['train_loss'].append(train_loss)
        results_dict['valid_loss'].append(np.mean(batch_loss))
        results_dict["learning_rate"].append(learning_rate)
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the dataframe to disk
        df.to_csv(f"{model_loc}/training_log.csv", index=False)

        # Stop training if we have not improved after X epochs (stopping patience)
        best_epoch = [i for i, j in enumerate(
            epoch_test_losses) if j == min(epoch_test_losses)][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break


if __name__ == '__main__':

    description = "Train a segmengation model on a hologram data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit {n_nodes} workers to PBS.",
    )
    parser.add_argument(
        "-m",
        dest="mode",
        type=str,
        default="none",
        help="If using more than 1 GPU, which mode to use (options are currently ddp or none).",
    )
    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    mode = args_dict.pop("mode")
    launch = bool(int(args_dict.pop("launch")))

    # ### Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Create directories if they do not exist and copy yml file
    os.makedirs(conf["save_loc"], exist_ok=True)
    if not os.path.exists(os.path.join(conf["save_loc"], "model.yml")):
        shutil.copy(config, os.path.join(conf["save_loc"], "model.yml"))

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if mode in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, mode)
    else:
        main(0, 1, conf, mode)
