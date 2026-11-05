# Solar Filament Instance Segmentation (v5 Pipeline)

An end-to-end deep learning pipeline for high-resolution instance segmentation of solar filaments from full-disk H$\alpha$ solar images using UNet, an ImageNet-pretrained ResNet34 encoder, recall-biased BCE-Tversky loss, overlapping sliding-window inference, and connected-component post-processing.

Developed for the **MAGFiLO 1.0 (Kaggle 2026)** competition.

---

## 🌟 Key Features in v5

1. **High-Resolution Overlapping Tiling ($2048 \times 2048 \rightarrow 512 \times 512$, Stride 384):**
   - Whole-disk downsampling crushes thin solar filaments and fine barb structures.
   - Tiles cover the full disk with a 25% overlap (128 px margin) to ensure zero boundary clipping.
2. **COCO Polygon Rasterization with Memory Management:**
   - Directly parses the official MAGFiLO annotations (`images` and `annotations` segmentation polygons).
   - Uses an in-memory LRU cache and automatic garbage collection to eliminate CPU RAM memory leaks during training.
3. **Pretrained Transfer Learning Encoder (ResNet34 + SMP):**
   - Employs an ImageNet-pretrained ResNet34 backbone via `segmentation_models_pytorch` for rich feature representation.
4. **Recall-Biased BCE-Tversky Loss ($\alpha=0.3, \beta=0.7$):**
   - Heavily penalizes false negatives ($\beta=0.7$) to resolve missed filaments and improve Recognition Quality ($RQ$).
5. **Early Stopping:**
   - Tracks validation Dice score with a patience of 3 epochs to prevent overfitting and save GPU compute.
6. **Contiguous Instance Separation & COCO RLE Counts Export:**
   - Post-processes blended sigmoid probability maps with 8-connectivity connected components (`cv2.connectedComponentsWithStats`) and an area threshold of 500 px to prevent over-segmenting elongated filaments.
   - Outputs valid COCO `RLE Counts` format via `pycocotools`.

---

## ⚙️ System & Environment Specifications

- **OS:** Windows 11 (64-bit)
- **Language:** Python `3.11`
- **Compute:** NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM)
- **CUDA:** `12.8` (Mixed Precision AMP enabled)
- **Host RAM:** 16 GB
- **Random Seed:** `42` (Fixed across Python, NumPy, PyTorch, and cuDNN for strict reproducibility)

---

## 📦 Dependencies

Install the core requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install segmentation-models-pytorch opencv-python pycocotools scikit-image scipy pandas numpy pillow tqdm matplotlib
```

Key library versions used:
- `torch`: `2.11.0+cu128`
- `torchvision`: `0.26.0+cu128`
- `segmentation-models-pytorch`: `0.5.0`
- `opencv-python`: `5.0.0`
- `pycocotools`: `2.0.11`

---

## 🔬 Hyperparameters & Pipeline Setup

| Parameter | Value | Description |
|---|---|---|
| **Full Resolution** | $2048 \times 2048$ | Full solar disk image canvas |
| **Tile Size / Stride** | $512 \times 512$ / $384$ | 25% sliding window overlap |
| **Empty Tile Filter** | Active ($5\%$ empty retained) | Keeps all filament tiles + 5% background sample |
| **Architecture** | UNet (SMP) | ResNet34 encoder (ImageNet weights) |
| **Optimizer** | `AdamW` | Learning rate: $1\times 10^{-4}$, Weight decay: $1\times 10^{-4}$ |
| **LR Scheduler** | `CosineAnnealingLR` | $T_{\max} = 20$, $\eta_{\min} = 1\times 10^{-6}$ |
| **Batch Size** | `8` | Fits comfortably in $\sim$6 GB VRAM |
| **Max Epochs** | `20` | Early stopping with patience $= 3$ |
| **Loss Function** | BCE-Tversky | $w_{\text{BCE}} = 0.5$, $\alpha = 0.3$, $\beta = 0.7$ |
| **Train / Val Split** | $85\% / 15\%$ | Split deterministically by image ID |
| **Prob Threshold** | `0.45` | Binarization cutoff on stitched probabilities |
| **Min Area Cutoff** | `500 px` | 8-connected components noise rejection |

---

## 📁 Repository Structure

```text
├── solar_filament_segmentationv5.ipynb   # Main end-to-end v5 pipeline notebook
├── README.md                             # Project documentation and reproduction guide
└── .gitignore                            # Excludes dataset, weights, and environment files
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure Dataset Path:**
   Open `solar_filament_segmentationv5.ipynb` and ensure `Config.BASE_DIR` points to your local MAGFiLO 1.0 dataset directory:
   ```python
   BASE_DIR = Path("path/to/MAGFiLO_1.0_Kaggle_2026")
   ```

4. **Execute the Notebook:**
   Run all cells sequentially in `solar_filament_segmentationv5.ipynb`:
   - **Cells 1–4:** Environment validation, configuration, and tiled dataset creation.
   - **Cell 5:** Exploratory data analysis (tile visualizer).
   - **Cells 6–7:** UNet model definition and training loop with early stopping.
   - **Cells 8–10:** Overlapping full-disk prediction, Panoptic Quality (PQ) validation evaluation, and test set inference.
   - **Output:** Generates `best_filament_unet_v5.pt` (best checkpoint) and `submission_v5.csv` (180 test images formatted in COCO RLE counts).

---

## 🔒 Data & Security Notice

In strict adherence to Kaggle competition rules and open-source best practices:
- No raw competition images (`train_images/`, `test_images/`) or private annotations (`.json`) are committed to this repository.
- Model weight files (`*.pt`, `*.pth`) and virtual environments (`.venv/`) are excluded via `.gitignore`.

