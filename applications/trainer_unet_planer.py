import warnings
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import functools
from echo.src.base_objective import BaseObjective

from argparse import ArgumentParser
from pathlib import Path

import torch.fft
import logging
import shutil
import os
import sys
import yaml

from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.checkpoint as DCP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
from holodec.unet import PlanerSegmentationModel as SegmentationModel
from holodec.planer_datasets import LoadHolograms, UpsamplingReader
from holodec.planer_trainer import Trainer
from holodec.pbs import launch_script, launch_script_mpi
from holodec.seed import seed_everything
from holodec.losses import load_loss
from holodec.transforms import LoadTransformations
from holodec.scheduler import load_scheduler

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_model_wrapper(conf, vae, device):

    if conf["trainer"]["mode"] == "fsdp":

        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100_000
        )

        model = FSDP(
            vae,
            use_orig_params=True,  # needed if using torch.compile
            auto_wrap_policy=auto_wrap_policy,
            # cpu_offload=CPUOffload(offload_params=True)
        )

    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(vae, device_ids=[device])
    else:
        model = vae

    return model


def load_model_and_optimizer(conf, model, device):

    start_epoch = conf['trainer']['start_epoch']
    save_loc = conf['save_loc']
    learning_rate = conf['trainer']['learning_rate']
    weight_decay = conf['trainer']['weight_decay']
    amp = conf['trainer']['amp']

    #  Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if start_epoch == 0:  # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    # load optimizer and grad scaler states
    else:
        ckpt = f"{save_loc}/checkpoint.pt"
        checkpoint = torch.load(ckpt, map_location=device)

        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")

            # wait for all workers to get the model loaded
            torch.distributed.barrier()

            # tell torch how we are loading the data in (e.g. sharded states)
            FSDP.set_state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
            )
            # different from ``torch.load()``, DCP requires model state_dict prior to loading to get
            # the allocated storage and sharding information.
            state_dict = {
                "model_state_dict": model.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model.load_state_dict(state_dict["model_state_dict"])

            # Load the optimizer here on all ranks
            # https://github.com/facebookresearch/fairscale/issues/1083
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            curr_opt_state_dict = checkpoint["optimizer_state_dict"]
            optim_shard_dict = model.get_shard_from_optim_state_dict(curr_opt_state_dict)
            # https://www.facebook.com/pytorch/videos/part-5-loading-and-saving-models-with-fsdp-full-state-dictionary/421104503278422/
            # says to use scatter_full_optim_state_dict
            optimizer.load_state_dict(optim_shard_dict)

        elif conf["trainer"]["mode"] == "ddp":
            logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        else:
            logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return model, optimizer, scheduler, scaler


def trainer(rank, world_size, conf, trial=False, distributed=False):

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])
        distributed = True

    # infer device id from rank

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    # Set up training and validation file names. Use the prefix to use style-augmented data sets
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]

    # Complete the dataset configuration with missing parameters
    train_transforms = LoadTransformations(conf["transforms"]["training"])
    valid_transforms = LoadTransformations(conf["transforms"]["validation"])

    # Create the UpsamplingReader using the updated configuration
    train_dataset = UpsamplingReader(
        conf,
        "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_training.nc",
        train_transforms
    )
    valid_dataset = UpsamplingReader(
        conf,
        "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc",
        valid_transforms
    )

    # setup the distributed sampler
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,  # May be True
            seed=seed,
            drop_last=True
        )

        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=False,
            drop_last=True
        )
    else:
        train_sampler = None
        valid_sampler = None

    # setup the dataloder for this process

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False if distributed else True,  # shuffling handled by a sampler
        sampler=train_sampler,
        pin_memory=False,
        num_workers=0,
        drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=False,
        num_workers=0,
        drop_last=True
    )

    # model
    vae = SegmentationModel(conf)

    num_params = sum(p.numel() for p in vae.parameters())
    if rank == 0:
        logging.info(f"Number of parameters in the model: {num_params}")
    # summary(vae, input_size=(channels, height, width))

    # have to send the module to the correct device first

    vae.to(device)
    # vae = torch.compile(vae)

    # Wrap in DDP or FSDP module, or none

    model = distributed_model_wrapper(conf, vae, device)

    # Load an optimizer, scheduler, and gradient scaler from disk if epoch > 0

    model, optimizer, scheduler, scaler = load_model_and_optimizer(conf, model, device)

    # Train and validation losses

    train_criterion = load_loss(conf["loss"]["training_loss_mask"], split="mask train")
    valid_criterion = load_loss(conf["loss"]["validation_loss_mask"], split="mask valid")
    # Set up some metrics

    metrics = {"dice": load_loss("dice", split="mask metric")}  # Mask metrics

    # Initialize a trainer object

    trainer = Trainer(model, rank, module=(conf["trainer"]["mode"] == "ddp"))

    # Fit the model

    result = trainer.fit(
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        trial=trial
    )

    return result


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            return trainer(0, 1, conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                raise E
                #raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


if __name__ == "__main__":

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
        help="Submit workers to PBS.",
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
    launch = int(args_dict.pop("launch"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

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

#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Derecho parallelism",
#         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
#         # track hyperparameters and run metadata
#         config=conf
#     )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        trainer(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        trainer(0, 1, conf)
