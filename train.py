import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # for showing progress bar in terminal

from src.data.dataset import MapDataset
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils import save_checkpoint, save_some_examples
# Na samej górze w importach w train.py dodajcie:
from src.augmentation import MILD_AUGMENTATION # Możecie użyć STRONG_AUGMENTATION, ale MILD na początek jest bezpieczniejsze
# hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_DISC = 1e-4
LEARNING_RATE_GEN = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100

def main():
    print(f"Starting on device: {DEVICE}")

    # initializing models and sending them to the GPU
    discriminator = Discriminator(in_channels=3).to(DEVICE)
    generator = Generator(in_channels=3).to(DEVICE)

    # preparing data (dataset and dataloader)
    dataset = MapDataset(root_dir="data/maps/train", augmentations=MILD_AUGMENTATION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Data prepared. There are {len(dataloader)} batches to work on")

    # Loss Functions
    # BCE judges how much model made a mistake in true / false grade
    bce_loss = nn.BCEWithLogitsLoss()
    # using WithLogits to change numbers to 0-1 before counting the error

    # L1 Loss (Mean Absolute Error): checks pixel after pixel how far is generated image from real one
    l1_loss = nn.L1Loss()

    #L1 weight
    L1_LAMBDA = 10

    # Optimizers
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))
    opt_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
    # parameter betas=(0.5, 0.999) makes chaotic training of GANs more stable

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{NUM_EPOCHS} ---")

        # putting dataloader in progress bar
        loop = tqdm(dataloader, leave=True)

        for idx, (x, y) in enumerate(loop):
            # x: sketch (condition), y: real image from satellite (goal)
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_fake = generator(x)

            # STEP A: training discriminator - goal: teach discriminator to give 1 for originals and 0 for generated (false)
            # looking at real pair at first
            D_origin = discriminator(x, y)
            # loss_D_origin = bce_loss(D_origin, torch.ones_like(D_origin)) # we want to have result 1 (True) -> ones_like
            loss_D_origin = bce_loss(D_origin, torch.ones_like(D_origin) * 0.9)
            # looking at fake pair (sketch + generated image)
            D_generated = discriminator(x, y_fake.detach()) # using detach for training only discriminator
            loss_D_generated = bce_loss(D_generated, torch.zeros_like(D_generated)) # we want to have result 0 (False) -> zeros_like

            loss_D = (loss_D_origin + loss_D_generated) / 2

            # clearing gradients
            discriminator.zero_grad() # removing old memory
            loss_D.backward() # backpropagation
            opt_discriminator.step()

            # STEP B: training generator - goal: forcing 1 on discriminator and learn to colour generated image
            D_generated2 = discriminator(x, y_fake) # without detach because we are training generator
            loss_G_generated = bce_loss(D_generated2, torch.ones_like(D_generated2)) # we want to force 1 -> ones_like
            loss_G_L1 = l1_loss(y_fake, y) * L1_LAMBDA # checking differences in colours between generated and original

            # total loss of generator
            loss_G = loss_G_generated + loss_G_L1

            # clearing gradients
            generator.zero_grad()
            loss_G.backward()
            opt_generator.step()

            # progress bar update
            loop.set_postfix(
                D_loss = loss_D.item(),
                G_loss = loss_G.item(),
            )

        # saving sample images in order to see how the model learns
        save_some_examples(generator, dataloader, epoch, folder="saved_images", device=DEVICE)

        # Zapis logów do pliku tekstowego
        with open("loss_log.txt", "a") as f:
            f.write(f"Epoch: {epoch+1}/{NUM_EPOCHS} | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}\n")
        
        # saving weights in the case of sudden stop of training
        save_checkpoint(
            model=generator,
            optimizer=opt_generator, 
            folder="checkpoints",
            filename=f"generator_epoch_{epoch+1:03d}.pth.tar"
            )
        save_checkpoint(
            model=discriminator,
            optimizer=opt_discriminator,
            folder="checkpoints",
            filename=f"discriminator_epoch_{epoch+1:03d}.pth.tar"
            )

# security check - allowing code to start only when calling the file
if __name__ == "__main__":
    main()