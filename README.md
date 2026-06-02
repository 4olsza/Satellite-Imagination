# Satellite-Imagination

**Map-to-Satellite Image Translation using Conditional GANs (Pix2PIx)...**

## 📋 Struktura Projektu

```
├── train.py                 # Główny skrypt treningu
├── config.py                # Konfiguracja hyperparametrów
├── dataset.py               # Dataset loader
├── dataset.py               # Dataset class
├── requirements.txt         # Zależności
├── src/
│   ├── __init__.py
│   ├── utilis.py           # MLOpsManager (checkpoints, examples)
│   ├── losses.py           # Loss functions (GAN + L1)
│   ├── visualize.py        # Funkcje do wizualizacji
│   └── models/
│       ├── __init__.py
│       ├── generator.py    # Generator (U-Net)
│       └── discriminator.py # Discriminator (PatchGAN)
└── data/
    └── train/              # Dane treningowe
```

## 🚀 Quick Start

### 1. Instalacja Zależności

```bash
pip install -r requirements.txt
```

### 2. Przygotowanie Danych

Umieść obrazy treningowe w `data/train/`. 
Każdy obraz powinien zawierać mapę po lewej i satelitę po prawej stronie (podzielone pionową linią).

```
data/
└── train/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Uruchomienie Treningu

```bash
python train.py
```

Trening będzie:
- Zapisywać checkpointy co 500 kroków w `./checkpoints/`
- Zapisywać przykłady co 100 kroków w `./examples/`
- Czyścić stare checkpointy (zachowując 5 ostatnich)

### 4. Wznowienie Treningu z Checkpointa

Trening automatycznie wznawia się z ostatniego checkpointa. Jeśli chcesz zacząć od nowa:

```bash
rm -rf checkpoints/
python train.py
```

## ⚙️ Konfiguracja

Edytuj `config.py`:

```python
from config import get_full_config

config = get_full_config()
config.batch_size = 32  # Zwiększ batch size
config.num_epochs = 200  # Trenuj dłużej
config.lambda_l1 = 50.0  # Mniej L1 loss
```

### Presets

```python
from config import get_small_config, get_full_config

# Szybki test
config = get_small_config()  # 4 batch, 5 epok

# Pełny trening
config = get_full_config()   # 16 batch, 100 epok
```

## 📊 Monitorowanie Treningu

### Checkpointy

```python
from src.utilis import MLOpsManager

mlops = MLOpsManager()
mlops.list_checkpoints()  # Wyświetl dostępne checkpointy
```

### Przykłady

Wygenerowane przykłady są zapisywane w `./examples/` co 100 kroków.
Format: `[Mapa | Rzeczywisty Satelita | Wygenerowany Satelita]`

## 🔧 Kluczowe Komponenty

### Generator (U-Net)
- Input: Mapa (256x256 RGB)
- Output: Satelita (256x256 RGB)
- Encoder-decoder z skip connections

### Discriminator (PatchGAN)
- Input: Mapa + Satelita (konkatenacja)
- Output: Patch-wise klasyfikacja
- Lepsze uczenie szczegółów lokalnych

### Loss Functions

1. **GAN Loss** (Binary Cross Entropy)
   - Zmusza generatora do tworzenia realistycznych obrazów

2. **L1 Reconstruction Loss** (λ=100)
   - Zmusza generatora do pixel-wise podobieństwa
   - Zapobiega rozmytym wynikiem

### MLOpsManager

- `save_checkpoint()` - Zapisuje model + optimizer state
- `load_checkpoint()` - Wczytuje model + optimizer state
- `save_some_examples()` - Zapisuje co 100 kroków
- `cleanup_old_checkpoints()` - Usuwa stare checkpointy

## 📈 Hyperparametry

| Parametr | Wartość | Opis |
|----------|---------|------|
| `learning_rate_gen` | 2e-4 | Learning rate generatora |
| `learning_rate_disc` | 2e-4 | Learning rate dyskryminatora |
| `lambda_l1` | 100.0 | Waga L1 reconstruction loss |
| `batch_size` | 16 | Rozmiar batcha |
| `num_epochs` | 100 | Liczba epok |
| `save_checkpoint_every` | 500 | Co ile kroków zapis checkpoint |
| `save_examples_every` | 100 | Co ile kroków zapis przykładów |

## 🐛 Troubleshooting

### Błąd: "Brak obrazów w folderze"
```
Upewnij się że:
1. Folder data/train/ istnieje
2. Zawiera obrazy (.jpg, .png, .jpeg)
3. Obrazy mają co najmniej 256x256 px
```

### Błąd: "Out of Memory"
```python
# Zmniejsz batch size w config.py
config.batch_size = 8  # zamiast 16
```

### Generator generuje szare/rozmyte obrazy
```
Zwiększ lambda_l1 lub trenuj dłużej
config.lambda_l1 = 200.0
config.num_epochs = 200
```

## 📚 Referencje

- Pix2Pix Paper: https://arxiv.org/abs/1611.05957
- PyTorch: https://pytorch.org/
- U-Net: https://arxiv.org/abs/1505.04597

## 👨‍💻 Autor

Maciej - Gałąź: `maciek`
