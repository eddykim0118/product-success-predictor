# RSNA Intracranial Aneurysm Detection

**[Kaggle Competition](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)**

Deep learning project for detecting and localizing intracranial aneurysms from 3D MRA/CTA brain vessel images.

## 🎯 Competition Overview

### Task
- **Multi-label Classification**: Predict 14 targets
  - `aneurysm_present`: Whether aneurysm exists (weight 13)
  - 13 location-specific labels (weight 1 each):
    - ICA (Internal Carotid Artery) - L/R
    - MCA (Middle Cerebral Artery) - L/R
    - ACA (Anterior Cerebral Artery) - L/R
    - PCA (Posterior Cerebral Artery) - L/R
    - PCOM (Posterior Communicating Artery) - L/R
    - BA (Basilar Artery)
    - VA (Vertebral Artery) - L/R

### Evaluation Metric
```
Final Score = (1/2) × (AUC_AP + (1/13) × Σ AUC_location)
```
- Weighted Mean AUC ROC
- `aneurysm_present` has weight 13 in evaluation

## 📁 Project Structure

```
rsna-intracranial-aneurysm-detection/
│
├── src/
│   ├── data/
│   │   ├── loader.py          # DICOM data loader
│   │   ├── preprocessor.py    # CT/MRA preprocessing
│   │   └── augmentation.py    # 3D data augmentation
│   │
│   ├── models/
│   │   ├── resnet3d.py        # 3D ResNet (10/18/34/50)
│   │   ├── unet3d.py          # 3D U-Net, Attention U-Net
│   │   └── ensemble.py        # Ensemble, TTA
│   │
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── metrics.py         # AUC, Dice, IoU, etc.
│   │   └── losses.py          # Focal, Asymmetric Loss, etc.
│   │
│   └── utils/
│       ├── visualization.py   # Image visualization
│       └── dicom_utils.py     # DICOM utilities
│
├── scripts/
│   ├── train.py              # Run training
│   ├── evaluate.py           # Run evaluation
│   ├── predict.py            # Run inference
│   └── visualize_results.py  # Visualize results
│
├── configs/
│   ├── baseline.yaml         # Basic configuration
│   └── advanced.yaml         # Advanced configuration
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
│
├── data/
│   ├── raw/                  # Raw DICOM data
│   └── processed/            # Preprocessed data
│
└── outputs/
    ├── checkpoints/          # Model checkpoints
    ├── logs/                 # Training logs
    └── submissions/          # Kaggle submission files
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/eddykim/rsna-intracranial-aneurysm-detection.git
cd rsna-intracranial-aneurysm-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data

```bash
# Setup Kaggle API credentials first (~/.kaggle/kaggle.json)
kaggle competitions download -c rsna-intracranial-aneurysm-detection
unzip rsna-intracranial-aneurysm-detection.zip -d data/raw/
```

### 3. Training

```bash
# Train with baseline config
python scripts/train.py --config configs/baseline.yaml

# Train with custom settings
python scripts/train.py \
    --config configs/baseline.yaml \
    --model resnet34 \
    --epochs 100 \
    --batch-size 4
```

### 4. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data-dir data/raw
```

### 5. Inference

```bash
python scripts/predict.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --data-dir data/raw/test \
    --output submission.csv \
    --tta
```

## 🛠️ Models

### 3D ResNet
- ResNet-10/18/34/50/101/152 variants
- Adapted for 3D volumetric input

### 3D U-Net
- Standard U-Net for segmentation
- Attention U-Net with attention gates

### Ensemble
- Model averaging
- Weighted ensemble
- Test-Time Augmentation (TTA)
- Multi-scale inference

## 📊 Key Features

### Preprocessing
- DICOM to NumPy conversion
- Isotropic resampling (1mm³)
- HU windowing for CTA
- Brain extraction and cropping

### Augmentation
- 3D rotation, flip
- Intensity shift and scale
- Gaussian noise and blur
- Elastic deformation

### Loss Functions
- Weighted BCE for competition metric
- Focal Loss for class imbalance
- Asymmetric Loss for multi-label

## 📈 Results

| Model | Val AUC | LB Score |
|-------|---------|----------|
| ResNet-18 | - | - |
| ResNet-50 | - | - |
| Ensemble | - | - |

## 🔧 Configuration

See `configs/baseline.yaml` for all available options:

```yaml
model:
  type: resnet18
  num_classes: 14
  dropout: 0.5

training:
  epochs: 100
  batch_size: 4
  lr: 1e-4
  
loss:
  type: weighted_multilabel
  aneurysm_weight: 13.0
```

## 📝 Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory recommended

## 📚 References

- [RSNA Competition Page](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- [3D ResNets for Action Recognition](https://arxiv.org/abs/1711.09577)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## 👤 Author

**Eddy Kim**
- Statistics & Data Science @ BYU
- ML Engineer @ HEAL USA

## 📄 License

MIT License
