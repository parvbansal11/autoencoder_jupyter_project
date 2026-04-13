# Image Compression & Reconstruction via Autoencoders

**Parv Bansal — AIMS, DTU**

A deep learning project implementing and benchmarking four autoencoder architectures for lossy image compression on the CIFAR-10 dataset. The pipeline maps images through an encoder into a compact latent representation, then reconstructs them via a decoder.

---

## Models

| Model | Architecture | Latent Dim |
|-------|-------------|------------|
| ANN-v1 | Fully connected, shallow | 32 |
| ANN-v2 | Fully connected, deep | 64 |
| CNN-v1 | Convolutional + Batch Norm | 32 |
| CNN-v2 | Convolutional + Batch Norm | 64 |

## Pipeline

```
Image → Encoder → Latent Space → Decoder → Reconstructed Image
```

## Dataset

CIFAR-10 — 60,000 32x32 colour images across 10 classes. Downloaded automatically via `torchvision.datasets`.

## Requirements

```bash
pip3 install torch torchvision matplotlib scikit-learn gdown tqdm pillow notebook
```

## Usage

Open `autoencoder_premium.ipynb` in Jupyter or VS Code and run all cells. Trained model weights, reconstruction grids, loss curves, and a compression ratio summary are saved automatically.

## Results

Outputs include reconstruction comparisons, interpolation in latent space, PCA of latent representations, and a final compression ratio report across all four models.

## Tech Stack

Python, PyTorch, torchvision, Matplotlib, scikit-learn, tqdm
