# Satellite-Imagination

Projekt Pix2Pix do przekształcania map w obrazy satelitarne.

## 📋 Struktura projektu

```
Satellite-Imagination/
├── README.md
├── LICENSE
├── requirements.txt
├── augmentation.py        # Augmentacje obrazów mapy + satelita
├── dataset.py             # Główny loader danych dla treningu
├── inference.py           # Skrypt inferencji generujący obrazy satelitarne
├── metrics.py             # Metryki ewaluacji wygenerowanych obrazów
├── README.md              # Ten plik
├── .gitignore
├── .venv/                 # Virtualenv (nie wersjonujemy)
├── data/
│   └── train/             # Dane treningowe
└── src/
    ├── visualize.py       # Prosta wizualizacja par obrazów
    └── data/
        └── dataset.py     # Dataset klasowy (mapa + satelita)
    └── models/
        └── generator.py   # Generator U-Net
```

> Uwaga: plik `preprocess.py` został usunięty, ponieważ loader danych `dataset.py` wystarcza dla gotowego zbioru obrazów.

## 🚀 Szybki start

1. Aktywuj środowisko Pythona:

```bash
source .venv/bin/activate
```

2. Zainstaluj wymagania:

```bash
pip install -r requirements.txt
```

3. Przygotuj dane w `data/train/`.
   - Każdy plik powinien zawierać parę obrazów:
   - lewa połowa: satelita
   - prawa połowa: mapa
   - szerokość musi być parzysta, wysokość co najmniej 256 px

4. Uruchom trening lub inferencję zgodnie z istniejącymi skryptami.

## 🧩 Dane treningowe

Format danych:

- Obraz wejściowy ma dwie części połączone poziomo.
- Lewa połowa to prawdziwy obraz satelitarny.
- Prawa połowa to odpowiadająca mu mapa/sketch.

Dataset loader w `dataset.py`:
- wczytuje obrazy RGB,
- dzieli je na dwie części,
- normalizuje do przedziału `[-1, 1]`.

## 🧠 Model

W projekcie jest:

- `src/models/generator.py` — generator w stylu U-Net,
- brak klasycznego dyskryminatora w kodzie głównym.

Generator:
- Wejście: mapa 256x256 RGB,
- Wyjście: obraz satelitarny 256x256 RGB,
- Dekoder używa `Upsample + Conv2d`, co redukuje artefakty "szachownicy".

## 🔧 Inferencja

Skrypt `inference.py`:
- ładuje wytrenowany checkpoint,
- przekształca pojedynczy obraz mapy lub cały folder,
- zapisuje wygenerowane obrazy do `./output/`.

Przykład uruchomienia:

```bash
python inference.py --checkpoint path/to/checkpoint.pth --input data/test/mapa.png --output output/
```

## 📊 Ewaluacja

Plik `metrics.py` udostępnia metryki:
- `psnr()`,
- `ssim()`,
- `mae()`,
- `mse()`.

Użycie metryk jest proste: porównaj rzeczywisty obraz satelitarny z wygenerowanym.

## 📌 Uwagi

- Jeśli Twoja baza jest już przygotowana i działa, nie musisz używać żadnego preprocessu.
- `dataset.py` sam poradzi sobie z podstawowym formatem pary obrazów.
- Jeśli później dodamy surowe dane, można przywrócić lub napisać prosty skrypt przygotowawczy.

## 👨‍💻 Autor

Maciej
