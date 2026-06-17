# Satellite Imagination

A implementation of a Conditional Generative Adversarial Network (cGAN) based on the Pix2Pix architecture, trained to generate realistic satellite imagery from simplified map sketches.

## Opis

Model uczy się transformować mapy w obrazy satelitarne. Używamy warunkowego GAN-a (cGAN) z generatorem U-Net i dyskryminatorem. Generator generuje obrazy, a dyskryminator uczy się je oceniać.

## Requirements

- Python 3.8+
- PyTorch with CUDA (or CPU)
- GPU (optional, but it significantly speeds up training)

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
