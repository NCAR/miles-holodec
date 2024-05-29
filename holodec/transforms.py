import torch
import random
import logging
import torchvision
import numpy as np


logger = logging.getLogger(__name__)


pre_waveprop_transformations = [
    'GaussianNoise',
    'AdjustBrightness',
    'GaussianBlur'
]


def LoadTransformations(transform_config: str):
    tforms = []
    if "Normalize" in transform_config:
        mode = transform_config["Normalize"]["mode"]
        tforms.append(Preprocess(mode))
    if "PlanerNormalize" in transform_config:
        mode = transform_config["PlanerNormalize"]["mode"]
        tforms.append(PlanerPreprocess(mode))
    if "ToTensor" in transform_config:
        tforms.append(ToTensor())
    if "GaussianNoise" in transform_config:
        rate = transform_config["GaussianNoise"]["rate"]
        noise = transform_config["GaussianNoise"]["noise"]
        if rate > 0.0:
            tforms.append(GaussianNoise(rate, noise))
    if "RandomVerticalFlip" in transform_config:
        rate = transform_config["RandomVerticalFlip"]["rate"]
        if rate > 0.0:
            tforms.append(RandVerticalFlip(rate))
    if "RandomHorizontalFlip" in transform_config:
        rate = transform_config["RandomVerticalFlip"]["rate"]
        if rate > 0.0:
            tforms.append(RandHorizontalFlip(rate))
    if "AdjustBrightness" in transform_config:
        rate = transform_config["AdjustBrightness"]["rate"]
        brightness = transform_config["AdjustBrightness"]["brightness_factor"]
        if rate > 0.0:
            tforms.append(AdjustBrightness(rate, brightness))
    if "GaussianBlur" in transform_config:
        rate = transform_config["GaussianBlur"]["rate"]
        k_sz = transform_config["GaussianBlur"]["kernel_size"]
        sigma = transform_config["GaussianBlur"]["sigma"]
        if rate > 0.0:
            tforms.append(GaussianBlur(rate, k_sz, sigma))
    if "Standardize" in transform_config:
        tforms.append(Standardize())
    #transform = transforms.Compose(tforms)
    return tforms


class RandVerticalFlip(object):
    def __init__(self, rate):
        logger.info(
            f"Loaded RandomVerticalFlip transformation with probability {rate}")
        self.rate = rate

    def __call__(self, sample):
        if torch.rand(1).item() < self.rate:
            sample['image'] = torch.flip(sample['image'], dims=[-1])
            if 'mask' in sample:
                sample['mask'] = torch.flip(sample['mask'], dims=[-1])
        return sample


class RandHorizontalFlip(object):
    def __init__(self, rate):
        logger.info(
            f"Loaded RandomHorizontalFlip transformation with probability {rate}")
        self.rate = rate

    def __call__(self, sample):
        if torch.rand(1).item() < self.rate:
            sample['image'] = torch.flip(sample['image'], dims=[-2])
            if 'mask' in sample:
                sample['mask'] = torch.flip(sample['mask'], dims=[-2])
        return sample


class Standardize(object):
    """Standardize image"""

    def __init__(self):
        logger.info(
            "Loaded Standardize transformation that rescales data so mean = 0, std = 1")

    def __call__(self, sample):
        image = sample['image']
        image = (image - image.mean()) / image.std()
        sample["image"] = image
        return sample


class Preprocess(object):
    """Normalize image"""

    def __init__(self, mode="norm"):
        if mode == "norm":
            logger.info(
                "Loaded Preprocess transformation that normalizes data in the range 0 to 1")
        if mode == "stan":
            logger.info(
                "Loaded Preprocess transformation that standardizes data into z-scores")
        if mode == "sym":
            logger.info(
                "Loaded Preprocess transformation that normalizes data in the range -1 to 1")
        if mode == "255":
            logger.info(
                "Loaded Preprocess transformation that normalizes data color channel by dividing by 255.0 and phase pi")
        self.mode = mode

    def __call__(self, sample):

        image = sample['image']  # .astype(np.float32)

        if self.mode == "norm":
            # Compute min and max separately for even and odd dimensions
            even_min = image[:, 0::2, :, :].min()
            odd_min = image[:, 1::2, :, :].min()

            even_max = image[:, 0::2, :, :].max()
            odd_max = image[:, 1::2, :, :].max()

            # Normalize even dimensions to [0, 1]
            image[:, 0::2, :, :] = (image[:, 0::2, :, :] - even_min) / (even_max - even_min)

            # Normalize odd dimensions to [0, 1]
            image[:, 1::2, :, :] = (image[:, 1::2, :, :] - odd_min) / (odd_max - odd_min)

        if self.mode == "stan":
            # Compute mean and std separately for even and odd dimensions
            even_mean = image[:, 0::2, :, :].mean()
            odd_mean = image[:, 1::2, :, :].mean()

            even_std = image[:, 0::2, :, :].std()
            odd_std = image[:, 1::2, :, :].std()

            # Subtract mean and divide by std for even dimensions
            image[:, 0::2, :, :] -= even_mean
            image[:, 0::2, :, :] /= even_std

            # Subtract mean and divide by std for odd dimensions
            image[:, 1::2, :, :] -= odd_mean
            image[:, 1::2, :, :] /= odd_std

        if self.mode == "sym":
            # Normalize to the range [-1, 1] separately for even and odd dimensions
            even_min = image[:, 0::2, :, :].min()
            even_max = image[:, 0::2, :, :].max()

            odd_min = image[:, 1::2, :, :].min()
            odd_max = image[:, 1::2, :, :].max()

            # Normalize even dimensions to [-1, 1]
            image[:, 0::2, :, :] = -1 + 2.0 * (image[:, 0::2, :, :] - even_min) / (even_max - even_min)

            # Normalize odd dimensions to [-1, 1]
            image[:, 1::2, :, :] = -1 + 2.0 * (image[:, 1::2, :, :] - odd_min) / (odd_max - odd_min)

        if self.mode == "255":
            # Normalize the first channel
            image[0] /= 255.0

            # Process remaining channels in pairs (evens: absolute values, odds: angles)
            for i in range(1, image.shape[0], 2):
                # Calculate absolute values for even channels
                image[i - 1] = image[i - 1] / 255.0

                # Normalize angles for odd channels
                image[i] = (1.0 + image[i] / torch.pi) / 2.0

        sample["image"] = image
        return sample


class PlanerPreprocess(object):
    """Normalize image"""

    def __init__(self, mode="norm"):
        if mode == "norm":
            logger.info(
                f"Loaded Normalize transformation that normalizes data in the range 0 to 1")
        if mode == "stan":
            logger.info(
                f"Loaded Normalize transformation that standardizes data into z-scores")
        if mode == "sym":
            logger.info(
                f"Loaded Normalize transformation that normalizes data in the range -1 to 1")
        if mode == "255":
            logger.info(
                f"Loaded Normalize transformation that normalizes data color channel by dividing by 255.0 and phase pi")
        self.mode = mode

    def __call__(self, sample):

        image = sample['image']  # .astype(np.float32)

        if self.mode == "norm":
            image -= image.min()
            image /= image.max()

        if self.mode == "stan":
            image -= image.mean()
            image /= image.std()

        if self.mode == "sym":
            image = -1 + 2.0*(image - image.min())/(image.max() - image.min())

        if self.mode == "255":
            #image /= 255.0
            image[0] /= 255.0
            if image.shape[0] > 1:
                image[1] = (1.0 + image[1] / np.pi) / 2.0

        sample["image"] = image
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        logger.info(
            "Loaded ToTensor transformation")

    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        if len(image.shape) == 2:
            image = image.reshape(1, image.shape[0], image.shape[1])
        sample["image"] = torch.from_numpy(image).float()
        return sample


class AdjustBrightness(object):

    def __init__(self, rate, brightness):
        logger.info(
            "Loaded AdjustBrightness transformation")
        self.rate = rate
        self.brightness = brightness

    def __call__(self, sample):
        if self.rate >= 1.0:
            image = sample['image']
            brightness = random.uniform(0.0, self.brightness)
            image = torchvision.transforms.functional.adjust_brightness(
                image, brightness
            )
            sample["image"] = image
        elif random.random() < self.rate:
            image = sample['image']
            image = torchvision.transforms.functional.adjust_brightness(
                image, self.brightness
            )
            sample["image"] = image
        return sample


class GaussianBlur(object):

    def __init__(self, rate, kernel_size, sigma):
        logger.info(
            "Loaded GaussianBlur transformation")
        self.rate = rate
        self.kernel_size = int(kernel_size)
        self.sigma = sigma

    def __call__(self, sample):
        if self.rate >= 1.0:
            image = sample['image'].unsqueeze(0)
            sigma = random.uniform(0.0, self.sigma)
            image = torchvision.transforms.functional.gaussian_blur(
                image,
                kernel_size=self.kernel_size,
                sigma=sigma
            )
            sample["image"] = image.squeeze(0)
        elif random.random() < self.rate:
            image = sample['image'].unsqueeze(0)
            sigma = random.uniform(0.0, self.sigma)
            image = torchvision.transforms.functional.gaussian_blur(
                image,
                kernel_size=self.kernel_size,
                sigma=sigma
            )
            sample["image"] = image.squeeze(0)
        return sample


class GaussianNoise(object):

    def __init__(self, rate, noise):
        logger.info(
            "Loaded GaussianNoise transformation")
        self.rate = rate
        self.noise = noise

    def __call__(self, sample):
        if self.rate >= 1.0:
            image = sample['image']
            noise = torch.FloatTensor(image.shape).uniform_(0.0, self.noise)
            noise = torch.normal(0, noise)
            image += noise.to(image.device)
            sample["image"] = image
        elif random.random() < self.rate:
            image = sample['image']
            noise = torch.normal(0, self.noise, size=image.shape)
            image += noise
            sample["image"] = image
        return sample
