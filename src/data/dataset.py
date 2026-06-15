# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import random
# import torchvision.transforms.functional as TF

# class MapDataset(Dataset):
#     def __init__(self, root_dir):
#         # saving path to folder with images
#         self.root_dir = root_dir

#         # creating list with names of all files in this dir
#         self.list_files = os.listdir(self.root_dir)

#         # definition of transforming every image
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ]) 

#     def __len__(self):
#         # returning number of the images
#         return len(self.list_files)
    
#     def __getitem__(self, index):
#         # getting file's name
#         img_file = self.list_files[index]

#         # getting full path
#         img_path = os.path.join(self.root_dir, img_file)

#         # opening the image using PIL (RGB ensures us of having three colour channels)
#         image = Image.open(img_path).convert('RGB')

#         # data from dataset maps is horizontally concatenated so we need to split it excatly in a half
#         # checking width and height of the image
#         width, height = image.size

#         # using crop function to cut the half of the image
#         # satellite image - usually left side
#         satellite_img = image.crop((0, 0, width // 2, height))
#         # map sketch - usually right side
#         map_img = image.crop((width // 2, 0, width, height))

#         # data augmentation: rotating both images with propability of 50% to make training more precise
#         if random.random() > 0.5:
#             # TF.hflip = horizontal flip (horizontal mirror image)
#             satellite_img = TF.hflip(satellite_img)
#             map_img = TF.hflip(map_img)

#         # putting both images through transforms
#         satellite_tensor = self.transform(satellite_img)
#         map_tensor = self.transform(map_img)

#         # returning data in the tuple - input at first (sketch), then goal (satellite)
#         return map_tensor, satellite_tensor

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
        
        # Przyjmujemy naszą zaawansowaną klasę augmentacji z augmentation.py
        self.augmentations = augmentations

        # Zostawiamy tu TYLKO zamianę na tensor i normalizację (matematyka)
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

        # Aplikujemy Wasze zaawansowane augmentacje (z augmentation.py)
        if self.augmentations is not None:
            # Zakładamy, że podano mapę i satelitę (is_train=True)
            map_img, satellite_img = self.augmentations(map_img, satellite_img, is_train=True)
        else:
            # Fallback, jeśli ktoś zapomni podać augmentacji (żeby kod się nie wysypał)
            map_img = TF.resize(map_img, [256, 256])  # type: ignore
            satellite_img = TF.resize(satellite_img, [256, 256])  # type: ignore

        # Zamiana obrazków na Tensory z zakresem [-1, 1]
        satellite_tensor = self.transform_tensor(satellite_img)
        map_tensor = self.transform_tensor(map_img)

        # input at first (sketch), then goal (satellite)
        return map_tensor, satellite_tensor