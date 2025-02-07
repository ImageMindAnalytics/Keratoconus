import torch
import numpy as np
from torchvision.transforms import functional as F

from monai.transforms import (        
    BorderPad,
    CenterSpatialCrop,
    EnsureChannelFirst,    
    Compose,      
    NormalizeIntensity,      
    RandAffine,       
    RandAxisFlip, 
    RandRotate,
    RandFlip,
    RandZoom,
    RepeatChannel,
    Resize,
    ScaleIntensityRange,
    ScaleIntensity,
    ToTensor, 
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandSpatialCrop,
    LoadImaged,        
    EnsureChannelFirstd,
    RandAxisFlipd,
    RandAffined,
    RandRotated,
    RandZoomd,
    Resized,
    Rotate90d,
    ScaleIntensityRanged,
    ToTensord
)

class CenterCropByLabel:
    def __init__(self, label_number, img_key="img", seg_key="seg",  padding=0):
        """
        Initializes the transform.

        Args:
            label_number (int): The label number to crop around.            
            img_key (str): The key in the dictionary for the image. Default is "img".
            seg_key (str): The key in the dictionary for the label map. Default is "seg".
            padding (int): Optional padding to add around the crop. Default is 0.
        """
        self.label_number = label_number        
        self.img_key = img_key
        self.seg_key = seg_key
        self.padding = padding

    def __call__(self, img_d):
        """
        Applies the transform.

        Args:
            img_d (dict): A dictionary containing the image and label map.
        
        Returns:
            torch.Tensor: The dictionary with cropped image and label map.
        """

        image = img_d[self.img_key]
        label_map = img_d[self.seg_key]
                          
        # Ensure image and label map are tensors
        if not isinstance(image, torch.Tensor) or not isinstance(label_map, torch.Tensor):
            raise ValueError("Both image and label_map must be torch tensors")

        
        label_map = label_map.squeeze(0)  # (H, W)
        # Find the coordinates of the specified label
        label_mask = (label_map == self.label_number).nonzero(as_tuple=False)
        
        if label_mask.size(0) == 0:
            return img_d
        
        min_coords = label_mask.min(dim=0)[0]
        max_coords = label_mask.max(dim=0)[0]

        # Calculate the bounding box
        center_y = (min_coords[0] + max_coords[0]) // 2
        center_x = (min_coords[1] + max_coords[1]) // 2
        max_edge = max(max_coords[0] - min_coords[0], max_coords[1] - min_coords[1]) + self.padding

        half_size = max_edge // 2

        # Calculate crop coordinates
        top = max(0, center_y - half_size)
        left = max(0, center_x - half_size)
        bottom = min(image.size(1), center_y + half_size)
        right = min(image.size(2), center_x + half_size)

        # Crop the image
        cropped_image = image[:, top:bottom, left:right]

        # Pad if necessary to make it a square
        pad_height = max_edge - (bottom - top)
        pad_width = max_edge - (right - left)
        if pad_height > 0 or pad_width > 0:
            padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
            cropped_image = F.pad(cropped_image, padding, fill=0)

        img_d[self.img_key] = cropped_image

        cropped_seg = label_map[top:bottom, left:right].unsqueeze(0)
        if pad_height > 0 or pad_width > 0:
            cropped_seg = F.pad(cropped_seg, padding, fill=0)
        
        img_d[self.seg_key] = cropped_seg

        return img_d
    
class TrainTransform:
    def __init__(self, img_key="img", seg_key="seg"):

        # image augmentation functions
        self.train_transform = Compose(
            [   
                ToTensord(keys=[img_key, seg_key]),
                EnsureChannelFirstd(keys=[img_key], channel_dim=-1),
                EnsureChannelFirstd(keys=[seg_key], channel_dim='no_channel'),
                ScaleIntensityRanged(keys=[img_key], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                RandAxisFlipd(keys=[img_key, seg_key], prob=0.5),
                CenterCropByLabel(label_number=2, img_key=img_key, seg_key=seg_key),
                RandRotated(keys=[img_key, seg_key], range_x=np.pi/6, prob=0.5),
                Resized(keys=[img_key, seg_key], spatial_size=(256, 256)),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)  
    
class EvalTransform:
    def __init__(self, img_key="img", seg_key="seg"):

        # image augmentation functions
        self.eval_transform = Compose(
            [   
                ToTensord(keys=[img_key, seg_key]),
                EnsureChannelFirstd(keys=[img_key], channel_dim=-1),
                EnsureChannelFirstd(keys=[seg_key], channel_dim='no_channel'),
                ScaleIntensityRanged(keys=[img_key], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                CenterCropByLabel(label_number=2, img_key=img_key, seg_key=seg_key),                
                Resized(keys=[img_key, seg_key], spatial_size=(256, 256)),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)
