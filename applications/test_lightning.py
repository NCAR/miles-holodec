import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.cli import OptimizerCallable
from lightning.pytorch.callbacks import RichProgressBar

import holodec.losses
from holodec.unet import PlanerSegmentationModel as SegmentationModel
from holodec.datasets import UpsamplingReader, XarrayReader
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
                 activation: str = None):
        
        super().__init__()
        self.save_hyperparameters(ignore=('training_loss', 'validation_loss'))
        conf = self.hparams
        print(conf)
        self.model = SegmentationModel({'model': conf})
        self.train_criterion = training_loss
        self.val_criterion = validation_loss
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        pred_mask = self.model(inputs)
        loss = self.train_criterion(pred_mask, target.float())
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        pred_mask = self.model(inputs)
        loss = self.val_criterion(pred_mask, target.float())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

class Transforms:
    def __init__(self, training: dict, validation: dict, inference: dict):
        self.training_config = training
        self.validation_config = validation
        self.inference_config = inference

class HolodecDataModule(L.LightningDataModule):
    def __init__(self,
                 train_dataset: str,
                 validation_dataset: str,
                 transforms: Transforms = None,
                 train_batch_size=16,
                 valid_batch_size=16):
        
        super().__init__()
        self.train_datapath = train_dataset
        self.valid_datapath = validation_dataset
        
        self.train_conf = transforms.training_config
        self.valid_conf = transforms.validation_config
        self.inference_conf = transforms.inference_config
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        self.train_transforms = LoadTransformations(self.train_conf)
        self.valid_transforms = LoadTransformations(self.valid_conf)
        
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
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.valid_batch_size)
    
    def test_dataloader(self):
        pass
    
    def predict_dataloader(self):
        pass
    
    def teardown(self, stage: str):
        pass
    
class HolodecCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser.add_argument('--save_info.save_loc', type=str, required=True, help='Top level save directory')
        # parser.add_argument('--save_info.exp_name', type=str, default=None, required=False, help="Name of the experiment subfolder")
        # parser.add_argument('--save_info.exp_version', type=str, default=None, required=False, help="Version number of this experiment")
        #parser.add_lightning_class_args(RichProgressBar, 'rich_progress_bar')
        pass
    
    #def before_instantiate_classes(self) -> None:
        #self.config_init['callbacks'].append(RichProgressBar())
    
def main():
    cli = HolodecCLI(model_class=PlanarLightningModel,
                     datamodule_class=HolodecDataModule,
                     seed_everything_default=1000,
                     auto_configure_optimizers=True)
    
if __name__ == "__main__":
    main()