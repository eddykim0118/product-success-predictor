"""
Data Augmentation
=================

3D data augmentation techniques for CT/MRA images.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


class Compose:
    """
    Compose multiple transforms together.
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, data: Dict) -> Dict:
        for t in self.transforms:
            data = t(data)
        return data


class RandomRotation3D:
    """
    Random rotation around specified axes.
    """
    
    def __init__(
        self,
        angle_range: Tuple[float, float] = (-15, 15),
        axes: Tuple[int, int] = (1, 2),
        prob: float = 0.5,
    ):
        """
        Args:
            angle_range: (min_angle, max_angle) in degrees
            axes: Rotation axes (default: Y-Z plane)
            prob: Probability of applying
        """
        self.angle_range = angle_range
        self.axes = axes
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() > self.prob:
            return data
        
        angle = np.random.uniform(*self.angle_range)
        
        volume = data['volume']
        rotated = ndimage.rotate(
            volume,
            angle,
            axes=self.axes,
            reshape=False,
            order=1,
            mode='nearest'
        )
        data['volume'] = rotated
        
        # Rotate mask if present
        if 'mask' in data and data['mask'] is not None:
            data['mask'] = ndimage.rotate(
                data['mask'],
                angle,
                axes=self.axes,
                reshape=False,
                order=0,
                mode='nearest'
            )
        
        return data


class RandomFlip3D:
    """
    Random flip along specified axes.
    """
    
    def __init__(
        self,
        axes: Tuple[int, ...] = (1, 2),
        prob: float = 0.5,
    ):
        """
        Args:
            axes: Axes to potentially flip
            prob: Probability of flipping each axis
        """
        self.axes = axes
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        volume = data['volume']
        mask = data.get('mask')
        
        for axis in self.axes:
            if np.random.random() < self.prob:
                volume = np.flip(volume, axis=axis)
                if mask is not None:
                    mask = np.flip(mask, axis=axis)
        
        data['volume'] = np.ascontiguousarray(volume)
        if mask is not None:
            data['mask'] = np.ascontiguousarray(mask)
        
        return data


class RandomIntensityShift:
    """
    Random intensity shift and scale.
    """
    
    def __init__(
        self,
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5,
    ):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() > self.prob:
            return data
        
        volume = data['volume']
        
        shift = np.random.uniform(*self.shift_range)
        scale = np.random.uniform(*self.scale_range)
        
        volume = volume * scale + shift
        volume = np.clip(volume, 0, 1)
        
        data['volume'] = volume
        return data


class GaussianNoise:
    """
    Add Gaussian noise.
    """
    
    def __init__(
        self,
        std_range: Tuple[float, float] = (0.01, 0.05),
        prob: float = 0.5,
    ):
        self.std_range = std_range
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() > self.prob:
            return data
        
        volume = data['volume']
        std = np.random.uniform(*self.std_range)
        noise = np.random.normal(0, std, volume.shape).astype(np.float32)
        volume = np.clip(volume + noise, 0, 1)
        
        data['volume'] = volume
        return data


class GaussianBlur3D:
    """
    Apply Gaussian blur.
    """
    
    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.5, 1.5),
        prob: float = 0.3,
    ):
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() > self.prob:
            return data
        
        volume = data['volume']
        sigma = np.random.uniform(*self.sigma_range)
        volume = ndimage.gaussian_filter(volume, sigma=sigma)
        
        data['volume'] = volume
        return data


class RandomCrop3D:
    """
    Random crop from 3D volume.
    """
    
    def __init__(self, crop_size: Tuple[int, int, int]):
        """
        Args:
            crop_size: Target crop size (D, H, W)
        """
        self.crop_size = crop_size
    
    def __call__(self, data: Dict) -> Dict:
        volume = data['volume']
        mask = data.get('mask')
        
        d, h, w = volume.shape
        cd, ch, cw = self.crop_size
        
        # Calculate valid ranges
        d_start = np.random.randint(0, max(1, d - cd + 1))
        h_start = np.random.randint(0, max(1, h - ch + 1))
        w_start = np.random.randint(0, max(1, w - cw + 1))
        
        # Crop
        data['volume'] = volume[
            d_start:d_start+cd,
            h_start:h_start+ch,
            w_start:w_start+cw
        ]
        
        if mask is not None:
            data['mask'] = mask[
                d_start:d_start+cd,
                h_start:h_start+ch,
                w_start:w_start+cw
            ]
        
        return data


class CenterCrop3D:
    """
    Center crop from 3D volume.
    """
    
    def __init__(self, crop_size: Tuple[int, int, int]):
        self.crop_size = crop_size
    
    def __call__(self, data: Dict) -> Dict:
        volume = data['volume']
        mask = data.get('mask')
        
        d, h, w = volume.shape
        cd, ch, cw = self.crop_size
        
        d_start = (d - cd) // 2
        h_start = (h - ch) // 2
        w_start = (w - cw) // 2
        
        data['volume'] = volume[
            d_start:d_start+cd,
            h_start:h_start+ch,
            w_start:w_start+cw
        ]
        
        if mask is not None:
            data['mask'] = mask[
                d_start:d_start+cd,
                h_start:h_start+ch,
                w_start:w_start+cw
            ]
        
        return data


class Resize3D:
    """
    Resize 3D volume to target size.
    """
    
    def __init__(self, target_size: Tuple[int, int, int], order: int = 1):
        """
        Args:
            target_size: Target size (D, H, W)
            order: Interpolation order
        """
        self.target_size = target_size
        self.order = order
    
    def __call__(self, data: Dict) -> Dict:
        volume = data['volume']
        mask = data.get('mask')
        
        zoom_factors = [
            self.target_size[i] / volume.shape[i]
            for i in range(3)
        ]
        
        data['volume'] = ndimage.zoom(volume, zoom_factors, order=self.order)
        
        if mask is not None:
            data['mask'] = ndimage.zoom(mask, zoom_factors, order=0)
        
        return data


class ElasticDeformation3D:
    """
    Elastic deformation for 3D volumes.
    """
    
    def __init__(
        self,
        alpha: float = 100,
        sigma: float = 10,
        prob: float = 0.3,
    ):
        """
        Args:
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            prob: Probability of applying
        """
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() > self.prob:
            return data
        
        volume = data['volume']
        shape = volume.shape
        
        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma
        ) * self.alpha
        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma
        ) * self.alpha
        dz = ndimage.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma
        ) * self.alpha
        
        # Create coordinate grids
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # Apply deformation
        indices = [
            np.clip(z + dz, 0, shape[0]-1),
            np.clip(y + dy, 0, shape[1]-1),
            np.clip(x + dx, 0, shape[2]-1),
        ]
        
        data['volume'] = ndimage.map_coordinates(
            volume, indices, order=1, mode='nearest'
        ).reshape(shape)
        
        if 'mask' in data and data['mask'] is not None:
            data['mask'] = ndimage.map_coordinates(
                data['mask'], indices, order=0, mode='nearest'
            ).reshape(shape)
        
        return data


class Normalize:
    """
    Normalize volume to zero mean and unit variance or [0, 1] range.
    """
    
    def __init__(self, method: str = 'minmax'):
        """
        Args:
            method: 'minmax' or 'zscore'
        """
        self.method = method
    
    def __call__(self, data: Dict) -> Dict:
        volume = data['volume'].astype(np.float32)
        
        if self.method == 'minmax':
            v_min, v_max = volume.min(), volume.max()
            if v_max - v_min > 0:
                volume = (volume - v_min) / (v_max - v_min)
        elif self.method == 'zscore':
            mean, std = volume.mean(), volume.std()
            if std > 0:
                volume = (volume - mean) / std
        
        data['volume'] = volume
        return data


class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.
    """
    
    def __init__(self, add_channel: bool = True):
        """
        Args:
            add_channel: Add channel dimension
        """
        self.add_channel = add_channel
    
    def __call__(self, data: Dict) -> Dict:
        import torch
        
        volume = data['volume']
        
        if self.add_channel:
            volume = volume[np.newaxis, ...]  # (1, D, H, W)
        
        data['volume'] = torch.from_numpy(volume.astype(np.float32))
        
        if 'mask' in data and data['mask'] is not None:
            mask = data['mask']
            if self.add_channel:
                mask = mask[np.newaxis, ...]
            data['mask'] = torch.from_numpy(mask.astype(np.float32))
        
        if 'label' in data:
            data['label'] = torch.tensor(data['label'], dtype=torch.long)
        
        return data


def get_train_transforms(
    crop_size: Tuple[int, int, int] = (64, 128, 128),
    augment: bool = True,
) -> Compose:
    """
    Get training transforms.
    
    Args:
        crop_size: Random crop size
        augment: Whether to apply augmentation
        
    Returns:
        Compose of transforms
    """
    transforms = [
        Normalize(method='minmax'),
    ]
    
    if augment:
        transforms.extend([
            RandomRotation3D(angle_range=(-15, 15), prob=0.5),
            RandomFlip3D(axes=(1, 2), prob=0.5),
            RandomIntensityShift(prob=0.5),
            GaussianNoise(prob=0.3),
            GaussianBlur3D(prob=0.2),
        ])
    
    transforms.extend([
        RandomCrop3D(crop_size) if augment else CenterCrop3D(crop_size),
        ToTensor(add_channel=True),
    ])
    
    return Compose(transforms)


def get_val_transforms(
    crop_size: Tuple[int, int, int] = (64, 128, 128),
) -> Compose:
    """
    Get validation/test transforms.
    
    Args:
        crop_size: Center crop size
        
    Returns:
        Compose of transforms
    """
    return Compose([
        Normalize(method='minmax'),
        CenterCrop3D(crop_size),
        ToTensor(add_channel=True),
    ])

