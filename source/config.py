"""
Konfigurasi Pipeline Pelatihan dan Inferensi
Model Klasifikasi Anomali Modul Surya (12 Kelas)
"""

import os
import sys
from pathlib import Path

# Daftarkan direktori runtime DLL Intel oneAPI / SYCL untuk PyTorch XPU di Windows
venv_dir = Path(sys.executable).parent.parent
for sub in [
    venv_dir / "Library" / "bin",
    venv_dir / "Lib" / "site-packages" / "intel_extension_for_pytorch" / "bin",
]:
    if sub.exists():
        try:
            os.add_dll_directory(str(sub))
            os.environ["PATH"] = str(sub) + ";" + os.environ.get("PATH", "")
        except Exception:
            pass

import torch

# Path direktori
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "infraredsolarmodules" / "2020-02-14_InfraredSolarModules" / "InfraredSolarModules"
METADATA_PATH = DATASET_DIR / "module_metadata.json"
IMAGES_DIR = DATASET_DIR / "images"

# Direktori keluaran model dan log
OUTPUT_DIR = BASE_DIR / "source" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "best_solar_model.pth"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"

# Daftar 12 Kelas Anomali (Terurut secara konsisten)
CLASS_NAMES = [
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Hyperparameter Pelatihan
IMG_HEIGHT = 112
IMG_WIDTH = 112
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 7  # Early stopping patience
RANDOM_SEED = 42

# Rasio Pembagian Data (Stratified Split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Konfigurasi Hardware (Prioritaskan Intel Arc GPU / XPU, lalu CUDA, lalu CPU)
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

NUM_WORKERS = 0  # 0 disarankan di Windows untuk stabilitas multiprocessing
