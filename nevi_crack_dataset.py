from pathlib import Path
from typing import Any, Callable, Optional
import torch
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import cv2
from torchvision import transforms
import torch

class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 csv,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) /image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        self.csv = csv
        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            a = int(np.ceil(len(self.image_list) * (1 - self.fraction)))
            if a%2 == 0:
              a = a
            else: 
              a = a+1
            if subset == "Train":

                self.image_names = self.image_list[:a]
                self.mask_names = self.mask_list[:a]
            else:
                self.image_names = self.image_list[
                    a:]
                self.mask_names = self.mask_list[
                    a:]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        csv_info = self.csv
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            a=str(image_file).split('/')[-1].split('.jpg')[0]
            b=csv_info[csv_info['file_name'] == (a+'.jpg')].index.tolist()[0]
            label = csv_info.iloc[b][1]
            
            transform = transforms.ToPILImage()
            transform2 = transforms.Compose([transforms.ToTensor()])

            if self.image_color_mode == "rgb":
            
                image = image.convert("RGB")

                # Split into 3 channels
#                r, g, b = image.split()
#                res = torch.Tensor(2,500,500)
#                res[0] = transform2(r)
#                res[1] = transform2(g)
                # Increase Reds
#                r = r.point(lambda i: i * 2)
                # Decrease Greens
#                g = g.point(lambda i: i * 0)
                # Decrease Blues
#                b = g.point(lambda i: i * 0)
                #image = r
                # Recombine back to RGB image
#                image = Image.merge('RGB', (r, g, b))
#                image = transform(res)
            
#                image = np.asarray(image)
#                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#                equalized = clahe.apply(gray)
#                image = transforms.ToPILImage()(equalized)
                
            elif self.image_color_mode == "grayscale":
                image = image_clahe.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            a=torch.zeros([3,500, 500], dtype=torch.float32)
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
                sample["mask"][sample["mask"]>0] = 1.0
            if label==1:a[0]=sample["mask"]
            elif label==2:a[1]=sample["mask"]
            elif label==3:a[2]=sample["mask"]
            else:
              raise (ValueError(
                  f"csv label {label} is not a valid input. Check csv or code."
              ))
            sample = {"image": image, "mask": a}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
            return sample
