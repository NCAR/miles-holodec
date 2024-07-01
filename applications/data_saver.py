import xarray as xr
import torch
import yaml
import gc

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from argparse import ArgumentParser
from holodec.transforms import LoadTransformations
from holodec.datasets import UpsamplingReader

SAMPLES_TO_SAVE = 10000
SAVE_FILE = '/glade/derecho/scratch/jboothe/holograms/upsampled_holograms_all.nc'
SAVE_FREQ = 1000

parser = ArgumentParser(description='Save upsampled data to disk')
parser.add_argument(
    "-c",
    dest="data_config",
    type=str,
    default=False,
    help="Path to the data configuration (yml) containing your inputs.",
)
args = parser.parse_args()
args_dict = vars(args)
config = args_dict.pop("data_config")

with open(config) as cf:
    conf = yaml.load(cf, Loader=yaml.FullLoader)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')

# Complete the dataset configuration with missing parameters
conf["training_data"]["device"] = device
conf["training_data"]["transform"] = LoadTransformations(conf["transforms"]["training"])
conf["training_data"]["output_lst"] = [torch.abs, torch.angle]

# Create the UpsamplingReader using the updated configuration
train_dataset = UpsamplingReader(**conf["training_data"])

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=64
    )

# Assuming you have a PyTorch dataset (e.g., `my_dataset`) with 10,000 samples
# data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=False)

progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

# Add a variable for each sample
# with xr.open_dataset(SAVE_FILE, mode='w') as ds_file:
with progress_bar as progress:
    task = progress.add_task("Saving...", total=SAMPLES_TO_SAVE)

    # Create an empty Xarray dataset
    ds = xr.Dataset()
    for i, batch in enumerate(train_loader):
        
        sample = batch[0]  # Assuming each batch contains one sample
        #print(sample.shape)
        ds[f'sample_{i}'] = xr.DataArray(sample.numpy())
            
        progress.update(task, completed=i + 1)
        
        if i >= SAMPLES_TO_SAVE:
            break
        
        if i % SAVE_FREQ == 0:
            ds.to_netcdf(SAVE_FILE, mode='a')
            ds.close()
            del ds
            gc.collect()
            ds = xr.Dataset()
        
ds.to_netcdf(SAVE_FILE, mode='a')
ds.close()
