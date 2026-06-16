import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # for showing progress bar in terminal
from metrics import ImageMetrics
import os

from src.data.dataset import MapDataset
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils import save_checkpoint, save_some_examples
from src.augmentation import MILD_AUGMENTATION, NO_AUGMENTATION 
# hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_DISC = 1e-4
LEARNING_RATE_GEN = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 500
DECAY_START = NUM_EPOCHS / 2
MAX_CHECKPOINTS = 5

def lr_lambda(epoch):
    # during first 100 epochs returns 1.0, after dropping linearly to 0.0

    return 1.0 - max(0, epoch - DECAY_START) / float(NUM_EPOCHS - DECAY_START)

def main():
    print(f"Starting on device: {DEVICE}")

    # initializing models and sending them to the GPU
    discriminator = Discriminator(in_channels=3).to(DEVICE)
    generator = Generator(in_channels=3).to(DEVICE)

    # preparing data (dataset and dataloader)
    dataset = MapDataset(root_dir="data/maps/train", augmentations=MILD_AUGMENTATION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # preparing validating data for early stopping    
    val_dataset = MapDataset(root_dir="data/maps/val", augmentations=NO_AUGMENTATION)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Data prepared. There are {len(dataloader)} batches to work on")

    # Loss Functions
    # BCE judges how much model made a mistake in true / false grade
    bce_loss = nn.BCEWithLogitsLoss()
    # using WithLogits to change numbers to 0-1 before counting the error

    # L1 Loss (Mean Absolute Error): checks pixel after pixel how far is generated image from real one
    l1_loss = nn.L1Loss()

    #L1 weight
    L1_LAMBDA = 90

    # Optimizers
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))
    opt_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
    # parameter betas=(0.5, 0.999) makes chaotic training of GANs more stable

    # Schedulers will take care of dropping lr
    scheduler_discriminator = optim.lr_scheduler.LambdaLR(opt_discriminator, lr_lambda)
    scheduler_generator = optim.lr_scheduler.LambdaLR(opt_generator, lr_lambda)

    # early stopping 
    best_val_metric = float('inf')  # minimalizing errors (MAE)
    patience = 15                   
    patience_counter = 0

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

        # saving logs to text file
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
        
        # removing old checkpoints to clean the disc
        if (epoch + 1) > MAX_CHECKPOINTS:
            old_epoch = (epoch) + 1 - MAX_CHECKPOINTS

            generator_path = os.path.join("checkpoints", f"generator_epoch_{old_epoch:03d}.pth.tar")
            discriminator_path = os.path.join("checkpoints", f"discriminator_epoch_{old_epoch:03d}.pth.tar")

            if os.path.exists(generator_path):
                os.remove(generator_path)

            if os.path.exists(discriminator_path):
                os.remove(discriminator_path)
        
        scheduler_discriminator.step()
        scheduler_generator.step()

        # Early stopping
        generator.eval()
        
        with torch.no_grad():
            val_map, val_sat = next(iter(val_loader))
            val_map, val_sat = val_map.to(DEVICE), val_sat.to(DEVICE)
            
            # generating test file
            generated_val = generator(val_map)
            
            # counting metrics from metrics.py
            current_metric = ImageMetrics.mae(val_sat, generated_val)
            
            print(f"=> Epoch {epoch+1} | Val MAE Metric: {current_metric:.4f}")

            # checking if its better
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0 
                
                # NADPISUJEMY "NAJLEPSZY" MODEL
                torch.save(generator.state_dict(), "best_generator.pth")
                print("=> New best model saved!")
            else:
                patience_counter += 1
                print(f"=> No improvement for {patience_counter} epochs.")

            # terminating training
            # if patience_counter >= patience:
                # print(f"EARLY STOPPING: Model stopped validationg learing for {patience_counter} epochs. Terminated the training!")
                # break
                
        generator.train()

        # showing the current learning rate
        current_lr = opt_generator.param_groups[0]['lr']
        print(f"-> Current learing rate fo generator: {current_lr:.6f}")

# security check - allowing code to start only when calling the file
if __name__ == "__main__":
    main()