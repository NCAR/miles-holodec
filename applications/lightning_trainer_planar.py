import lightning as L
import torch
import os
import scipy.ndimage
import numpy as np

from torch.utils.data import DataLoader
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.cli import OptimizerCallable
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import BasePredictionWriter

import holodec.losses
from holodec.unet import PlanerSegmentationModel as SegmentationModel
from holodec.datasets import UpsamplingReader, XarrayReader
from holodec.planer_datasets import LoadHolograms
from holodec.planer_transforms import LoadTransformations


class PlanarLightningModel(L.LightningModule):
    def __init__(self,
                 training_loss: torch.nn.Module = holodec.losses.DiceBCELoss(),
                 validation_loss: torch.nn.Module = holodec.losses.DiceLoss(),
                 name: str = None,
                 encoder_name: str = None,
                 encoder_weights: str = None,
                 in_channels: int = 0,
                 classes: int = 0,
                 activation: str = None,
                 prediction_threshold: float = 0.5):
        
        super().__init__()
        self.save_hyperparameters(ignore=('training_loss', 'validation_loss', 'prediction_threshold'))
        conf = self.hparams
        self.model = SegmentationModel({'model': conf})
        self.train_criterion = training_loss
        self.val_criterion = validation_loss
        self.prediction_threshold = prediction_threshold
        
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch        
        pred_mask = self.model(inputs)
        loss = self.train_criterion(pred_mask, target.float())
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        pred_mask = self.model(inputs)
        loss = self.val_criterion(pred_mask, target.float())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs, target, h_idx, z_idx = batch
        pred_prob = self.model(inputs)
        # apply thresh
        pred_mask = pred_prob > self.prediction_threshold
        # scipy cluster
        pred_coordinates = []
        if pred_mask.sum() > 0:
            arr, n = scipy.ndimage.label(pred_mask.squeeze().cpu().numpy())
            _centroid = scipy.ndimage.find_objects(arr)
            # compute (x,y,d)
            for particle in _centroid:
                xind = (particle[0].stop + particle[0].start) // 2
                yind = (particle[1].stop + particle[1].start) // 2
                dind = max([
                    abs(particle[0].stop - particle[0].start), 
                    abs(particle[1].stop - particle[1].start)
                ])
                pred_coordinates.append([h_idx.item(), xind, yind, z_idx.item(), dind])
        # repeat for target
        true_coordinates = []
        if target.sum() > 0:
            arr, n = scipy.ndimage.label(target.squeeze().cpu().numpy())
            _centroid = scipy.ndimage.find_objects(arr)
            for particle in _centroid:
                xind = (particle[0].stop + particle[0].start) // 2
                yind = (particle[1].stop + particle[1].start) // 2
                dind = max([
                    abs(particle[0].stop - particle[0].start), 
                    abs(particle[1].stop - particle[1].start)
                ])
                true_coordinates.append([h_idx.item(), xind, yind, z_idx.item(), dind])
                
        return pred_coordinates, true_coordinates


class Transforms:
    def __init__(self, training: dict, validation: dict, predict: dict):
        self.training_config = training
        self.validation_config = validation
        self.predict_config = predict


class HolodecDataModule(L.LightningDataModule):
    def __init__(self,
                 train_dataset: str = '',
                 validation_dataset: str = '',
                 predict_dataset: str = '',
                 transforms: Transforms = None,
                 train_batch_size=16,
                 valid_batch_size=16,
                 predict_batch_size=16,
                 num_workers=8):
        
        super().__init__()
        self.train_datapath = train_dataset
        self.valid_datapath = validation_dataset
        self.predict_datapath = predict_dataset
        self.num_workers = num_workers
        
        self.train_conf = transforms.training_config
        self.valid_conf = transforms.validation_config
        self.predict_conf = transforms.predict_config
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.predict_batch_size = predict_batch_size
        
        self.train_transforms = LoadTransformations(self.train_conf)
        self.valid_transforms = LoadTransformations(self.valid_conf)
        self.predict_transforms = LoadTransformations(self.predict_conf)
        
    def prepare_data(self):
        # Called on single process only
        pass
    
    def setup(self, stage: str = None):
        # Called on every GPU
        if stage == 'fit' or stage is None:
            self.train_dataset = XarrayReader(
                self.train_datapath,
                self.train_transforms,
                mode="mask"
            )
            self.valid_dataset = XarrayReader(
                self.valid_datapath,
                self.valid_transforms,
                mode="mask"
            )
        if stage == 'predict':
            self.predict_dataset = LoadHolograms(
                self.predict_datapath,
                n_bins=1000,
                transform=self.predict_transforms
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size)
    
    def test_dataloader(self):
        pass
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
            )
    
    def teardown(self, stage: str):
        pass
    

class HolodecWriter(BasePredictionWriter):
    
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
    
    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pred_coords, true_coords = prediction
        pred_array = np.array(pred_coords)
        true_array = np.array(true_coords)
        
        fname_p = os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.csv")
        with open(fname_p, "ab") as f:
            np.savetxt(f, pred_array, delimiter=',')
            # f.write(b"\n")
            
        fname_t = os.path.join(self.output_dir, f"true_{trainer.global_rank}.csv")
        with open(fname_t, "ab") as f:
            np.savetxt(f, true_array, delimiter=',')
            # f.write(b"\n")
        
    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ):
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))
    
    
class HolodecCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser.add_argument('--save_info.save_loc', type=str, required=True, help='Top level save directory')
        # parser.add_argument('--save_info.exp_name', type=str, default=None, required=False, help="Name of the experiment subfolder")
        # parser.add_argument('--save_info.exp_version', type=str, default=None, required=False, help="Version number of this experiment")
        # parser.add_lightning_class_args(RichProgressBar, 'rich_progress_bar')
        # parser.add_argument('--mode', choices=['fit'], default='fit', help='Choose trainer mode, such as fit or predict')
        pass
    
    def before_fit(self):
        # DO FSDP Config Changes here!
        # self.config = *change whatever*
        pass
    
    # def before_instantiate_classes(self) -> None:
        # self.config_init['callbacks'].append(RichProgressBar())


def main():
    cli = HolodecCLI(model_class=PlanarLightningModel,
                     datamodule_class=HolodecDataModule,
                     seed_everything_default=1000,
                     auto_configure_optimizers=True) # run=False
    #cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
