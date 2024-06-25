import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.cli import OptimizerCallable
from lightning.pytorch.callbacks import RichProgressBar

import holodec.losses
from holodec.unet import SegmentationModel
from holodec.datasets import UpsamplingReader, XarrayReader
from holodec.planer_transforms import LoadTransformations

class LightningModel(L.LightningModule):
    def __init__(self,
                 training_loss_mask: torch.nn.Module = holodec.losses.DiceBCELoss(),
                 training_loss_depth: torch.nn.Module = holodec.losses.DiceBCELoss(),
                 training_loss_depth_mult: float = 1,
                 validation_loss_mask: torch.nn.Module = holodec.losses.DiceLoss(),
                 validation_loss_depth: torch.nn.Module = holodec.losses.DiceLoss(),
                 validation_loss_depth_mult: float = 1,
                 metric_mask: torch.nn.Module = holodec.losses.DiceLoss(),
                 metric_depth: torch.nn.Module = torch.nn.L1Loss(),
                 name: str = None,
                 encoder_name: str = None,
                 encoder_weights: str = None,
                 in_channels: int = 0,
                 classes: int = 0,
                 activation: str = None):
        
        super().__init__()
        self.save_hyperparameters('name', 'encoder_name', 'encoder_weights',
                                  'in_channels', 'classes', 'activation')
        conf = self.hparams
        print(conf)
        self.model = SegmentationModel({'model': conf})
        self.train_criterion = [training_loss_mask, training_loss_depth]
        self.val_criterion = [validation_loss_mask, validation_loss_depth]
        self.training_loss_depth_mult = training_loss_depth_mult
        self.validation_loss_depth_mult = validation_loss_depth_mult
        self.metric_criterion = [metric_mask, metric_depth]
    
    def on_training_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/mask_metric": 0, "hp/depth_metric": 0})
    
    def training_step(self, batch, batch_idx):
        inputs, y_part_mask, y_depth_mask, y_weight_mask = batch
        y_pred = self.model(inputs)
        y_pred_mask = y_pred[:, 0]
        y_pred_depth = y_pred[:, 1]
        
        mask_loss = self.train_criterion[0](y_part_mask, y_pred_mask)
        depth_loss = self.train_criterion[1](y_depth_mask, y_pred_depth, y_part_mask, y_pred_mask)
        loss = mask_loss + (depth_loss * self.training_loss_depth_mult)
        
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_depth_loss', depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        mask_metric_val = self.metric_criterion[0](y_part_mask, y_pred_mask)
        depth_metric_val = self.metric_criterion[1](y_depth_mask * y_part_mask, y_pred_depth * y_part_mask)
        self.log('metric/mask_train', mask_metric_val, on_step=True, on_epoch=True, logger=True)
        self.log('metric/depth_train', depth_metric_val, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, y_part_mask, y_depth_mask, y_weight_mask = batch
        y_pred = self.model(inputs)
        y_pred_mask = y_pred[:, 0]
        y_pred_depth = y_pred[:, 1]
        
        mask_loss = self.val_criterion[0](y_part_mask, y_pred_mask)
        depth_loss = self.val_criterion[1](y_depth_mask, y_pred_depth, y_part_mask, y_pred_mask)
        loss = mask_loss + (depth_loss * self.training_loss_depth_mult)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mask_loss', mask_loss, on_epoch=True, logger=True)
        self.log('val_depth_loss', depth_loss, on_epoch=True, logger=True)
        
        mask_metric_val = self.metric_criterion[0](y_part_mask, y_pred_mask)
        depth_metric_val = self.metric_criterion[1](y_depth_mask * y_part_mask, y_pred_depth * y_part_mask)
        
        self.log('hp/mask_metric', mask_metric_val, on_epoch=True)
        self.log('hp/depth_metric', depth_metric_val, on_epoch=True)
        return loss


class Transforms:
    def __init__(self, training: dict, validation: dict, inference: dict):
        self.training_config = training
        self.validation_config = validation
        self.inference_config = inference


def get_gpu_string(local_rank: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    else:
        return "cpu"


class HolodecDataModule(L.LightningDataModule):
    def __init__(self,
                 lookahead: int,
                 transforms: Transforms = None,
                 training_data: dict = None,
                 validation_data: dict = None,
                 inference_data: dict = None,
                 train_batch_size: int = 16,
                 valid_batch_size: int = 16):
        
        super().__init__()
        self.lookahead = lookahead
        self.train_config = training_data
        self.valid_config = validation_data
        self.inference_config = inference_data
        self.transforms = transforms
    
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        self.train_config["transform"] = LoadTransformations(self.transforms.training_config)
        self.valid_config["transform"] = LoadTransformations(self.transforms.validation_config)
        # self.inference_config["transform"] = LoadTransformations(self.transforms.inference_config)
        
        self.train_config["output_lst"] = [torch.abs, torch.angle]
        self.valid_config["output_lst"] = [torch.abs, torch.angle]
        # self.inference_config["output_lst"] = [torch.abs, torch.angle]
        
        self.train_config["lookahead"] = self.lookahead
        self.valid_config["lookahead"] = self.lookahead
        # self.inference_config["lookahead"] = self.lookahead
        
    def prepare_data(self):
        # Called on single process only
        pass
    
    def setup(self, stage: str = None):
        # Called on every GPU
        if stage == 'fit' or stage is None:
            device = get_gpu_string(self.trainer.local_rank)
            print(device)
            self.train_config["device"] = device
            self.valid_config["device"] = device
            
            self.train_dataset = UpsamplingReader(**self.train_config)
            self.valid_dataset = UpsamplingReader(**self.valid_config)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size
        )
    
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
        # parser.add_lightning_class_args(RichProgressBar, 'rich_progress_bar')
        # parser.add_argument('--mode', choices=['fit'], default='fit', help='Choose trainer mode, such as fit or predict')
        parser.link_arguments(
            source='data.lookahead',
            target='model.in_channels',
            compute_fn=lambda x: 2 * x,
            apply_on='parse'
        )
        # parser.link_arguments(
        #     source='data.lookahead',
        #     target='data.training_data.lookahead',
        #     apply_on='instantiate'
        # )
        # parser.link_arguments(
        #     source='data.lookahead',
        #     target='data.valid_data.lookahead',
        #     apply_on='instantiate'
        # )
        # parser.link_arguments(
        #     source='data.lookahead',
        #     target='data.inference_data.lookahead'
        # )
        pass
    
    def before_fit(self):
        # DO FSDP Config Changes here!
        # self.config = *change whatever*
        pass
    
    #def before_instantiate_classes(self) -> None:
        #self.config_init['callbacks'].append(RichProgressBar())


def main():
    cli = HolodecCLI(model_class=LightningModel,
                     datamodule_class=HolodecDataModule,
                     seed_everything_default=1000,
                     auto_configure_optimizers=True) # run=False
    #cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
