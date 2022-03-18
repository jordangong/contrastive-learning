from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])
    return color_distort


class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)


class RandomGaussianBlur(object):
    """
        PyTorch version of
        https://github.com/google-research/simclr/blob/244e7128004c5fd3c7805cf3135c79baa6c3bb96/data_util.py#L311
    """

    def gaussian_blur(self, image, sigma):
        image = image.reshape(1, 3, 224, 224)
        radius = np.int(self.kernel_size / 2)
        kernel_size = radius * 2 + 1
        x = np.arange(-radius, radius + 1)

        blur_filter = np.exp(
            -np.power(x, 2.0) / (2.0 * np.power(np.float(sigma), 2.0)))
        blur_filter /= np.sum(blur_filter)

        conv1 = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), groups=3,
                                padding=[kernel_size // 2, 0], bias=False)
        conv1.weight = torch.nn.Parameter(torch.Tensor(np.tile(
            blur_filter.reshape(kernel_size, 1, 1, 1), 3
        ).transpose([3, 2, 0, 1])))

        conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), groups=3,
                                padding=[0, kernel_size // 2], bias=False)
        conv2.weight = torch.nn.Parameter(torch.Tensor(np.tile(
            blur_filter.reshape(kernel_size, 1, 1, 1), 3
        ).transpose([3, 2, 1, 0])))

        res = conv2(conv1(image))
        assert res.shape == image.shape
        return res[0]

    def __init__(self, kernel_size, sigma_range=(0.1, 2), p=0.5):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            if np.random.uniform() < self.p:
                return self.gaussian_blur(
                    img, sigma=np.random.uniform(*self.sigma_range)
                )
            return img


class MultiCropDatasetWrapper(Dataset):
    """
    Modified from Facebook SwAV at: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/src/multicropdataset.py#L18
    """

    def __init__(
            self,
            dataset: Dataset,
            n_crops: list[int],
            crop_sizes: list[tuple[int, int]],
            crop_scale_ranges: list[tuple[float, float]],
            other_transforms: Optional[transforms.Compose] = None,
    ):
        assert len(crop_sizes) == len(n_crops)
        assert len(crop_scale_ranges) == len(n_crops)

        if hasattr(dataset, 'transform') and dataset.transform is not None:
            raise AttributeError('Please pass the transform to wrapper.')

        self.dataset = dataset

        trans = []
        for crop_size, crop_scale_range, n_crop in zip(
                crop_sizes, crop_scale_ranges, n_crops
        ):
            rand_resize_crop = transforms.RandomResizedCrop(
                crop_size, scale=crop_scale_range
            )
            if other_transforms is not None:
                trans_i = transforms.Compose([
                    rand_resize_crop, other_transforms
                ])
            else:
                trans_i = rand_resize_crop
            trans += [trans_i] * n_crop
        self.transform = trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        multi_crops = list(map(lambda trans: trans(img), self.transform))

        return multi_crops, target
