"""
Modul Dataset Loader, Transformasi Augmentasi, dan Stratified Splitter
untuk Citra Termal Modul Surya (InfraredSolarModules)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from config import (
    CLASS_TO_IDX,
    DATASET_DIR,
    IMAGES_DIR,
    IMAGE_SIZE,
    METADATA_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


class SolarModuleDataset(Dataset):
    """PyTorch Dataset untuk citra inframerah modul panel surya."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        images_dir: Path,
        transform=None,
    ):
        """
        Args:
            samples: List pasangan (nama_file_gambar, class_id)
            images_dir: Path direktori folder citra
            transform: Transformasi torchvision
        """
        self.samples = samples
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_name, label = self.samples[idx]
        image_path = self.images_dir / image_name

        # Buka gambar sebagai Grayscale ('L') lalu duplikasi ke 3 channel (RGB)
        # agar kompatibel dengan pre-trained weight model CNN (ImageNet)
        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = transforms.ToTensor()(img)

        return image_tensor, label


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Mendefinisikan pipeline transformasi augmentasi untuk pelatihan dan evaluasi."""
    # Mean dan Std normalisasi ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transforms, val_test_transforms


def load_and_split_data(
    metadata_path: Path = METADATA_PATH,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Membaca metadata JSON dan membagi dataset menjadi Train, Validation, dan Test
    menggunakan Stratified Split untuk menjamin proporsi 12 kelas seimbang di setiap split.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"File metadata tidak ditemukan di: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata: Dict[str, dict] = json.load(f)

    file_names = []
    labels = []

    for img_id, item in metadata.items():
        rel_path = item["image_filepath"]
        file_name = Path(rel_path).name
        cls_name = item["anomaly_class"]

        if cls_name in CLASS_TO_IDX:
            file_names.append(file_name)
            labels.append(CLASS_TO_IDX[cls_name])

    # 1. Split Train vs (Val + Test)
    val_test_combined_ratio = VAL_RATIO + TEST_RATIO
    train_files, val_test_files, train_labels, val_test_labels = train_test_split(
        file_names,
        labels,
        test_size=val_test_combined_ratio,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # 2. Split Val vs Test
    test_sub_ratio = TEST_RATIO / val_test_combined_ratio
    val_files, test_files, val_labels, test_labels = train_test_split(
        val_test_files,
        val_test_labels,
        test_size=test_sub_ratio,
        stratify=val_test_labels,
        random_state=RANDOM_SEED,
    )

    train_samples = list(zip(train_files, train_labels))
    val_samples = list(zip(val_files, val_labels))
    test_samples = list(zip(test_files, test_labels))

    return train_samples, val_samples, test_samples


def compute_class_weights(
    train_samples: List[Tuple[str, int]],
    num_classes: int = NUM_CLASSES,
    smoothing_power: float = 0.5,
) -> torch.Tensor:
    """
    Menghitung bobot kelas berimbang yang dihaluskan (Square-Root / Power Dampened Weights)
    untuk mencegah penalti yang terlalu ekstrem pada kelas mayoritas (No-Anomaly)
    sembari tetap memberi dorongan representasi pada anomali langka.
    Rumus: weight[c] = (total_samples / (num_classes * count[c])) ^ smoothing_power
    """
    labels = [sample[1] for sample in train_samples]
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)

    raw_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    dampened_weights = np.power(raw_weights, smoothing_power)

    # Normalisasi bobot agar mean(weights) = 1.0 untuk kestabilan gradient
    weights = dampened_weights / np.mean(dampened_weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Menghasilkan DataLoader siap pakai untuk Train, Val, Test beserta Bobot Kelas."""
    train_samples, val_samples, test_samples = load_and_split_data()
    train_transforms, val_transforms = get_data_transforms()

    train_dataset = SolarModuleDataset(train_samples, IMAGES_DIR, transform=train_transforms)
    val_dataset = SolarModuleDataset(val_samples, IMAGES_DIR, transform=val_transforms)
    test_dataset = SolarModuleDataset(test_samples, IMAGES_DIR, transform=val_transforms)

    class_weights = compute_class_weights(train_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights
