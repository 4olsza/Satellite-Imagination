import torch
import torchvision
import os

# CHECKPOINT FUNCTION: saving progress after each epoch for ability to resume training
def save_checkpoint(model, optimizer, folder="checkpoints", filename="checkpoint.pth.tar"):
    # creating folder if it doesn't exist yet
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    print(f"=> Saving weights to {filepath}")

    checkpoint = {
        "state_dict": model.state_dict(), # state_dict => dictionary of all weights learned by now
        "optimizer": optimizer.state_dict(), # saving state of optimizer in order to smoothly restart the training
    }

    torch.save(checkpoint, filepath)

# SAVING RESULTS FUNCTION: saving photos during training to check quality of generated images
def save_some_examples(generator, dataloader, epoch, folder="saved_images", device="cuda"):
    print("\n=> Saving sample preview images")

    # creating folder if one doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    generator.eval() # switching generator to evaluation in order to stop training

    # getting images and sending them to GPU
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        y_generated = generator(x)

        # reversal of normalization
        y_generated = y_generated * 0.5 + 0.5
        x_input = x * 0.5 + 0.5
        y_original = y * 0.5 + 0.5

        # saving images
        torchvision.utils.save_image(x_input[0], f"{folder}/epoch_{epoch+1:03d}_1_sketch.png")
        torchvision.utils.save_image(y_generated[0], f"{folder}/epoch_{epoch+1:03d}_2_generated.png")
        torchvision.utils.save_image(y_original[0], f"{folder}/epoch_{epoch+1:03d}_3_original.png")

    # switching to training mode again
    generator.train()