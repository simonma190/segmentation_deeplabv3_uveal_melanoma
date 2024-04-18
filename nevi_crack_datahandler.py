from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from UM_c3.nevi_crack_dataset import SegmentationDataset


def get_dataloader_sep_folder(data_dir: str = 'images',
                              image_folder: str = 'nevium_JPEGImages_resize_square',
                              mask_folder: str = 'nevium_SegmentationClass_resize_square',
                              batch_size: int = 2):
    """ Create Train and Test dataloaders from two
        separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Test
        ------Image
        ---------Image1
        ---------ImageM
        ------Mask
        ---------Mask1
        ---------MaskM
    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               transforms=data_transforms,
                               image_folder=image_folder,
                               mask_folder=mask_folder)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
#                      num_workers=8)
#                      num_workers=1
                      )                      
        for x in ['Train', 'Test']
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str = 'images',
                                 image_folder: str = 'img',
                                 mask_folder: str = 'gt',
                                 fraction: float = 0.2,
                                 batch_size: int = 2):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([
        #transforms.Resize((536, 356)),
        transforms.ToTensor(),
        
        ])

    

    df = pd.read_csv('/home/simon/data/sample2.csv')
    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms,
                               csv=df)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
#                      num_workers=8)
#                      num_workers=1
                      )                      
        for x in ['Train', 'Test']
    }
    return dataloaders
