import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class MapDataset(Dataset):
    def __init__(self, root_dir, augmentations=None):
        # saving path to folder with images
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        
        # taking augmentation class from augmentation.py
        self.augmentations = augmentations

        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) 

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)

        # RGB ensures us of having three colour channels
        image = Image.open(img_path).convert('RGB')

        width, height = image.size

        # satellite image - usually left side
        satellite_img = image.crop((0, 0, width // 2, height))
        # map sketch - usually right side
        map_img = image.crop((width // 2, 0, width, height))

        # adding augmentations
        if self.augmentations is not None:
            # assumed that map and satellite were given (is_train = True)
            map_img, satellite_img = self.augmentations(map_img, satellite_img, is_train=True)
        else:
            # if augmentation was not given
            map_img = TF.resize(map_img, [256, 256])  # type: ignore
            satellite_img = TF.resize(satellite_img, [256, 256])  # type: ignore

        # changing pictures to tensors [-1, 1]
        satellite_tensor = self.transform_tensor(satellite_img)
        map_tensor = self.transform_tensor(map_img)

        # input at first (sketch), then goal (satellite)
        return map_tensor, satellite_tensor