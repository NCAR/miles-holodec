import segmentation_models_pytorch as smp
import torch
import logging
import copy
import os
import torch.distributed.checkpoint as DCP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = logging.getLogger(__name__)


supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus
}

supported_encoders = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4',
                      'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100', 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100', 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l']


def load_model(model_conf):
    model_conf = copy.deepcopy(model_conf)
    name = model_conf.pop("name")
    if name in supported_models:
        logger.info(f"Loading model {name} with settings {model_conf}")
        return supported_models[name](**model_conf)
    else:
        raise OSError(
            f"Model name {name} not recognized. Please choose from {supported_models.keys()}")


class SegmentationModel(torch.nn.Module):

    def __init__(self, conf):

        super(SegmentationModel, self).__init__()

        in_channels = int(2 * conf["training_data"]["lookahead"])

        if conf['model']['name'] == 'unet':
            conf['model']['decoder_attention_type'] = 'scse'
        conf['model']['in_channels'] = in_channels
        conf['model']['classes'] = 2 # Mask, Depth
        self.activations = torch.nn.ModuleList([torch.nn.Sigmoid(), torch.nn.Identity()])
        self.model = load_model(conf['model'])

    def forward(self, x):
        x = self.model(x)
        return torch.cat([act(x[:, i:i+1, :, :]) for i, act in enumerate(self.activations)], dim=1)
        #return self.model(x)

    @classmethod
    def load_model(cls, conf):
        save_loc = conf['save_loc']
        ckpt = f"{save_loc}/checkpoint.pt" if conf["trainer"]["mode"] != "ddp" else f"{save_loc}/checkpoint_cuda:0.pt"

        if not os.path.isfile(ckpt):
            raise ValueError(
                "No saved checkpoint exists. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a model with pre-trained weights from path {ckpt}"
        )

        checkpoint = torch.load(ckpt)

        model_class = cls(**conf["model"])

        if conf["trainer"]["mode"] == "fsdp":
            FSDP.set_state_dict_type(
                model_class,
                StateDictType.SHARDED_STATE_DICT,
            )
            state_dict = {
                "model_state_dict": model_class.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model_class.load_state_dict(state_dict["model_state_dict"])
        else:
            model_class.load_state_dict(checkpoint["model_state_dict"])

        return model_class

    def save_model(self, conf):
        save_loc = conf['save_loc']
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, f"{save_loc}/checkpoint.pt")


class PlanerSegmentationModel(torch.nn.Module):

    def __init__(self, conf):

        super(PlanerSegmentationModel, self).__init__()

        if conf['model']['name'] == 'unet':
            conf['model']['decoder_attention_type'] = 'scse'
        conf['model']['in_channels'] = 1
        conf['model']['classes'] = 1 # Mask
        self.activations = torch.nn.ModuleList([torch.nn.Sigmoid(), torch.nn.Identity()])
        self.model = load_model(conf['model'])

    def forward(self, x):
        x = self.model(x)
        return x

    @classmethod
    def load_model(cls, conf):
        save_loc = conf['save_loc']
        ckpt = f"{save_loc}/checkpoint.pt" if conf["trainer"]["mode"] != "ddp" else f"{save_loc}/checkpoint_cuda:0.pt"

        if not os.path.isfile(ckpt):
            raise ValueError(
                "No saved checkpoint exists. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a model with pre-trained weights from path {ckpt}"
        )

        checkpoint = torch.load(ckpt)

        model_class = cls(**conf["model"])

        if conf["trainer"]["mode"] == "fsdp":
            FSDP.set_state_dict_type(
                model_class,
                StateDictType.SHARDED_STATE_DICT,
            )
            state_dict = {
                "model_state_dict": model_class.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model_class.load_state_dict(state_dict["model_state_dict"])
        else:
            model_class.load_state_dict(checkpoint["model_state_dict"])

        return model_class

    def save_model(self, conf):
        save_loc = conf['save_loc']
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, f"{save_loc}/checkpoint.pt")
