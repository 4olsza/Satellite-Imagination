import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # for showing progress bar in terminal

from src.data.dataset import MapDataset
from src.models.generator import Generator
from src.models.discriminator import Discriminator

# hiperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100

def main():
    print(f"Starting on device: {DEVICE}")

    # initialazing models and sending them to the graphic card
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, out_channels=3).to(DEVICE)

    # preparing data (dataset and dataloader)
    dataset = MapDataset(root_dir="data/maps/train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Data prepared. There are {len(dataloader)} batches to work on")

# security chech - allowing code to start only when calling the file
if __name__ == "__main__":
    main()

