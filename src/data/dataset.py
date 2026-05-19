import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MapDataset(Dataset):
    def __init__(self, root_dir):
        # saving path to folder with images
        self.root_dir = root_dir

        # creating list with names of all files in this dir
        self.list_files = os.listdir(self.root_dir)

        # definition of transforming every image
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) 

    def __len__(self):
        # returning number of the images
        return len(self.list_files)
    
    def __getitem__(self, index):
        # getting file's name
        img_file = self.list_files[index]

        # getting full path
        img_path = os.path.join(self.root_dir, img_file)

        # opening the image using PIL (RGB ensures us of having three colour channels)
        image = Image.open(img_path).convert('RGB')

        # data from dataset maps is horizontally concatenated so we need to split it excatly in a half
        # checking width and height of the image
        width, height = image.size

        # using crop function to cut the half of the image
        # satellite image - usually left side
        satellite_img = image.crop((0, 0, width // 2, height))
        # map sketch - usually right side
        map_img = image.crop((width // 2, 0, width, height))

        # putting both images through transforms
        satellite_tensor = self.transform(satellite_img)
        map_tensor = self.transform(map_img)

        # returning data in the tuple - input at first (sketch), then goal (satellite)
        return map_tensor, satellite_tensor