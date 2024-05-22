
from torchvision import datasets
from typing import Any, Tuple, Optional, Callable
from PIL import Image

class InfoCIFAR10(datasets.CIFAR10):
    def __init__(self,
                 root: str, 
                 train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
    
    def __getitem__(self, item: tuple) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index, scaler = item[0], item[1]
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, scaler

class InfoCIFAR100(datasets.CIFAR100):
    def __init__(self,
                 root: str, 
                 train: bool = True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
    
    def __getitem__(self, item: tuple) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index, scaler = item[0], item[1]
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, scaler