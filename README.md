# Satellite Imagination

*An implementation of a Conditional Generative Adversarial Network (cGAN) based on the Pix2Pix architecture, trained to generate realistic satellite imagery from simplified map sketches.*

## Project Overview

This model learns to translate abstract map layouts into photorealistic satellite images. It utilizes a Conditional GAN (cGAN) framework consisting of two competing neural networks:
* **The Generator (U-Net):** Learns to synthesize satellite images from map sketches. It uses skip connections to preserve essential spatial details and prevent the loss of low-level information.
* **The Discriminator (PatchGAN):** Evaluates the realism of the generated images. Instead of judging the entire image at once, it penalizes structure at the scale of local image patches, ensuring high-frequency details, sharp edges, and crisp textures.

Through adversarial training, the Generator continuously improves its ability to "fool" the Discriminator, resulting in highly accurate and realistic terrain generation.

## Key Features & Engineering Optimizations

This project features several advanced optimizations to ensure training stability, high-quality image synthesis, and hardware efficiency:
* **Bilinear Upsampling + Conv2D:** Replaced the standard transposed convolutions (`ConvTranspose2d`) with a combination of bilinear upsampling and 3x3 convolutions. This completely eliminated checkerboard artifacts, producing smooth color transitions.
* **Learning Rate Decay:** Integrated a linear learning rate scheduler (`LambdaLR`) that smoothly decays the learning rate to zero during the second half of the training process, allowing the model to finely chisel high-frequency terrain details.
* **Rolling Checkpoints (SSD Protection):** Implemented an automated disk cleanup routine that continuously stores only the 5 most recent training epochs, preventing the storage folder from consuming hundreds of gigabytes of disk space.
* **Synchronized Augmentation (Mild Augmentation):** Isolated the augmentation pipeline to apply identical random crops and horizontal flips simultaneously to both the map sketch and the target satellite image, while excluding rotations to prevent black triangular margin artifacts.

## Requirements & Installation

**Prerequisites:**
* Python 3.8+
* PyTorch with CUDA support (GPU is highly recommended for training speed, though CPU works for inference)

**Setup Instructions:**
1. Clone the repository to your local machine:
```bash
git clone https://github.com/4olsza/Satellite-Imagination.git
cd Satellite-Imagination
```
2. Install the required dependecies. We recommend using a virtual environment:
```bash
pip install -r requirements.txt
```

## Structure

```text
├── checkpoints/              # [gitignore] Automatically managed model weights (keeps max 5 recent epochs)
├── data/                     # [gitignore] Training dataset directory (downloaded separately)
│   └── maps/
│       └── train/
│       └── val/
├── saved_images/             # [gitignore] Preview of generated image samples saved after every epoch
├── src/
│   ├── data/
│   │   └── dataset.py        # Dataset class handling cropping and [-1, 1] normalization
│   ├── models/
│   │   ├── generator.py      # Generator architecture (U-Net with Bilinear modification)
│   │   └── discriminator.py  # Discriminator architecture (PatchGAN)
│   ├── augmentation.py       # Safe, synchronized data augmentation class
│   ├── metrics.py            # Evaluation math (PSNR, SSIM, MAE, MSE)
│   └── utils.py              # Helper functions for saving weights and image samples
├── tests/                    # Unit tests and debugging scripts
│   └── test_dataset.py       # Script to verify data loading and saving correctness
├── inference.py              # Script for running model inference on new sketches
├── loss_log.txt              # [gitignore] Lightweight text file logging the loss history
├── requirements.txt          # Tailored environment dependencies
└── train.py                  # Main training loop script
```
## Dataset preparation

This project uses the official `maps` dataset from the original Pix2Pix paper (University of California, Berkeley). Due to its size, the dataset is not included in this repository.

To prepare the data for training:
1. Go to the official Berkeley dataset repository: [Pix2Pix Datasets Index](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
2. Find and click on **`maps.tar.gz`** (approx. 239 MB) to download the archive.
3. Extract the downloaded archive.
4. Place the extracted files into the `data/maps/` directory so that the structure matches the following:

```text
data/
└── maps/
    ├── train/      # Used for training the model
    └── val/        # Used for validation/testing
```

## Training the Model

To start training the cGAN from scratch, run the main training script. The script is configured with optimal hyperparameters (e.g., 500 epochs with linear decay starting at epoch 250).

```bash
python train.py
```

During training:
* The model will automatically save intermediate generated images in the `saved_images/` directory after every epoch.
* Loss values will be logged continuously into `loss_log.txt`.
* The `checkpoints/` folder will automatically manage your disk space by keeping only the 5 most recent model weights.

## Inference

Once the model is trained, you can use it to translate new map sketches into satellite images. Ensure you have the trained weights in your `checkpoints/` folder. The script supports both single-image translation and batch processing via command-line arguments.

**For a single image:**
```bash
python inference.py --checkpoint checkpoints/generator_latest.pth --input data/maps/val/1.jpg --output ./output
```
**For a batch of images (processing an entire directory):**
```bash
python inference.py --checkpoint checkpoints/generator_latest.pth --input data/maps/val/ --output ./output
```

Available arguments:
* `--checkpoint` (required): Path to the trained model weights.
* `--input` (required): Path to a single map image or a directory of images.
* `--output` (optional): Directory where generated satellites will be saved (default: ./output).
* `--device` (optional): Set to cuda or cpu (default: cuda).

## Instalacja

```bash
git clone https://github.com/4olsza/Satellite-Imagination.git
cd Satellite-Imagination
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Format danych

Dane treningowe to obrazy ze złączonymi poziomo parami:
- **Lewa połowa**: obraz satelitarny
- **Prawa połowa**: mapa/sketch
- **Rozmiar**: co najmniej 256 px wysokości, szerokość parzysta (będą przeskalowane do 256x256)
- **Ścieżka**: `data/maps/train/`

Przykład:
```
data/maps/train/
├── mapa1.png    (512x256: satelita | mapa)
├── mapa2.png    (512x256: satelita | mapa)
└── ...
```

## Sposób działania

1. **Trening**: Generator uczy się generować realizm poprzez dwie straty:
   - Strata adversarialna (BCE): czy dyskryminator je rozróżni
   - Strata L1: zgodność pixel-po-pixelu z obrazem rzeczywistym

2. **Inferencja**: Wytrenowany generator transformuje mapę w obraz satelitarny

3. **Architektura**:
   - Generator: U-Net z skip connections (encoder-decoder)
   - Dyskryminator: PatchGAN (ocenia fragmenty obrazu)

## Uruchomienie

### Trening

```bash
python train.py
```

Parametry (w `train.py`):
- Learning rate: 2e-4
- Batch size: 16
- Epoki: 100
- L1 loss weight: 150

Zapisuje:
- Checkpointy: `checkpoints/generator_epoch_*.pth.tar`
- Przykładowe wygenerowane obrazy: `saved_images/`

### Inferencja

```bash
python inference.py --checkpoint checkpoints/generator_epoch_050.pth.tar --input map.png --output results/
```

Generuje obrazy satelitarne z map (pojedyncze zdjęcie lub folder).

### Testy

```bash
python test.py
```

Sprawdza poprawność wczytywania danych i modelu.

## Możliwości

- Generowanie realistycznych obrazów satelitarnych z map
- Ewaluacja metrykami: PSNR, SSIM, MAE, MSE (w `metrics.py`)
- Augmentacja danych (rotacje, flip, color jitter)
- Wizualizacja par obraz-mapa (skrypt `visualize.py`)
- Training z checkpointami co epokę

## Performance

- GPU (RTX 3070): ~15-20 min/epokę
- Pamięć: ~8GB VRAM
- CPU: znacznie wolniej (nie rekomendujemy)

## Uwagi

- Model zaktualizowany: dekoder bez artefaktów dzięki Upsample+Conv2d zamiast ConvTranspose2d
- Wymaga dużego, różnorodnego zbioru treningowego (mode collapse ze małymi zbiorami)
- Learning rate wrażliwy - domyślny 2e-4 jest wytunowany empirycznie

## Autorzy

Maciej, Krzysztof
