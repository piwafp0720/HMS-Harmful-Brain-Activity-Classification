import random

import albumentations as A
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as Tv2
from albumentations.pytorch.transforms import ToTensorV2

from .audio_transforms import Compose, TimeFreqMasking


class HMSAugmentations:
    def __init__(
        self,
        ver="ver_1",
    ):
        self.ver = ver
        self.spec_transform_val = A.Compose([], p=1.0)
        self.audio_transform_val = Compose([])
        self.spec_transform_test = A.Compose([], p=1.0)
        self.audio_transform_test = Compose([])

        if self.ver == "ver_2":
            self.spec_transform_train = A.Compose(
                [
                    TimeFreqMasking(
                        time_drop_width=32,
                        time_stripes_num=2,
                        freq_drop_width=16,
                        freq_stripes_num=2,
                        p=0.5,
                    )
                ],
                p=1.0,
            )
            self.audio_transform_train = Compose([])
