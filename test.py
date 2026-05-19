import matplotlib.pyplot as plt
from src.data.dataset import MapDataset

def test():
    # linking dataset
    dataset = MapDataset(root_dir="data/maps/train")

    # checking if __len__ method is working correctly
    print(f"Success! {len(dataset)} images founded in the training folder!")

    # taking first pair of the images
    map_tensor, satellite_tensor = dataset[0]

    # reversal of the normalization - using equation: (tensor * 0.5) + 0.5
    map_img = map_tensor * 0.5 + 0.5
    satellite_img = satellite_tensor * 0.5 + 0.5

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # moving channels to the correct order for matplotlib using permute function:
    #   [channels, height, width] -> [height, width, channels]
    axes[0].imshow(map_img.permute(1, 2, 0))
    axes[0].set_title("Map sketch")
    axes[0].axis("off")

    axes[1].imshow(satellite_img.permute(1, 2, 0))
    axes[1].set_title("Satellite image")
    axes[1].axis("off")

    # show
    plt.tight_layout()
    plt.show()

# security - starting test() function only when this file is called
if __name__ == "__main__":
    test()