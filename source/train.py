"""
Script Pelatihan Model Klasifikasi Anomali Modul Surya (12 Kelas)
Dilengkapi Dukungan Intel Extension for PyTorch (IPEX), Penanganan Class Imbalance,
Cosine Scheduler, Early Stopping berbasis Macro F1-Score, dan Evaluasi Lengkap.
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path
import numpy as np

# Abaikan peringatan deprecasi internal IPEX/PyTorch agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Cek ketersediaan Intel Extension for PyTorch (IPEX)
try:
    import intel_extension_for_pytorch as ipex

    HAS_IPEX = True
except ImportError:
    ipex = None
    HAS_IPEX = False

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DEVICE,
    LEARNING_RATE,
    METRICS_PATH,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import get_dataloaders
from model import build_model


def set_seed(seed: int = RANDOM_SEED):
    """Menyetel random seed untuk reproduksibilitas eksperimen."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[float, float]:
    """Satu epoch pelatihan dengan opsi Mixed Precision Bfloat16 (Intel IPEX / CPU AMP)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training Batch", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Gunakan Mixed Precision AMP jika bfloat16 aktif (didukung optimal di Intel Core Ultra / XPU)
        if use_bf16:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping untuk kestabilan gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[float, float, float, list, list]:
    """Evaluasi model pada data validation atau test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if use_bf16:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    eval_loss = running_loss / total
    eval_acc = correct / total
    eval_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return eval_loss, eval_acc, eval_macro_f1, all_targets, all_preds


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline 12-Class Solar Module Defect Classification")
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        choices=["compact_cnn", "mobilenetv3", "efficientnet", "resnet18", "resnet50"],
        help="Pilihan arsitektur model: compact_cnn (~450K), mobilenetv3 (~2.5M), efficientnet (~4M), resnet18 (~11M), resnet50 (~25M)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Ukuran batch")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--no_pretrained", action="store_true", help="Jangan gunakan pre-trained ImageNet weights")
    parser.add_argument("--use_ipex", action="store_true", default=True, help="Aktifkan Intel Extension for PyTorch jika tersedia")
    parser.add_argument("--bf16", action="store_true", help="Aktifkan Bfloat16 Mixed Precision di CPU Intel/XPU")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "xpu", "cuda"], help="Pilihan hardware: auto, cpu, xpu, atau cuda")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # Deteksi perangkat: CPU / XPU / CUDA
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "xpu":
        device = torch.device("xpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:  # auto
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print("=" * 70)
    print("  MEMULAI PIPELINE PELATIHAN KLASIFIKASI ANOMALI PANEL SURYA (12 KELAS)")
    print("=" * 70)
    print(f"Perangkat Komputasi : {device}")
    print(f"Arsitektur Model    : {args.model_type.upper()}")
    print(f"Jumlah Epochs       : {args.epochs}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Learning Rate       : {args.lr}")
    print(f"IPEX Extension      : {'Tersedia & Aktif' if (HAS_IPEX and args.use_ipex) else ('Tidak Terpasang' if not HAS_IPEX else 'Dinonaktifkan')}")
    print(f"Precision           : {'Bfloat16 (AMP)' if args.bf16 else 'Float32 (FP32)'}")
    print("-" * 70)

    # 1. Menyiapkan Data Loader dan Class Weights
    print("-> Membaca metadata dan membagi dataset (Stratified Split)...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
    )
    print(f"   Jumlah Data Latih (Train) : {len(train_loader.dataset)} sampel")
    print(f"   Jumlah Data Validasi (Val): {len(val_loader.dataset)} sampel")
    print(f"   Jumlah Data Uji (Test)    : {len(test_loader.dataset)} sampel")

    # Injeksi bobot kelas ke Cross Entropy Loss dengan Label Smoothing untuk regularisasi
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # 2. Inisialisasi Model
    print(f"-> Menginisialisasi arsitektur {args.model_type.upper()}...")
    model = build_model(
        model_type=args.model_type,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Optimasi IPEX (Intel Extension for PyTorch) jika diaktifkan
    if HAS_IPEX and args.use_ipex and device.type == "xpu":
        print("-> Mengoptimalkan Model & Optimizer dengan Intel Extension for PyTorch (XPU Operator Fusion)...")
        dtype = torch.bfloat16 if args.bf16 else torch.float32
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
        print("   [OK] Optimasi IPEX XPU berhasil diterapkan.")
    elif device.type == "cpu":
        print("-> [Info] Berjalan di CPU. Menggunakan akselerasi Intel oneDNN native bawaan PyTorch.")
    elif args.use_ipex and not HAS_IPEX:
        print("-> [Info] IPEX tidak terpasang. Pelatihan berjalan dengan PyTorch CPU standar (MKL-DNN).")

    # 5. Training Loop dengan Early Stopping berbasis Macro F1
    best_val_macro_f1 = 0.0
    patience_counter = 0

    print("\n-> Memulai Loop Pelatihan:")
    print(f"{'Epoch':<8} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val Macro F1':<13} | {'Durasi'}")
    print("-" * 80)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_bf16=args.bf16
        )
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model, val_loader, criterion, device, use_bf16=args.bf16
        )
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        print(
            f"{epoch:<8} | "
            f"{train_loss:<11.4f} | "
            f"{train_acc * 100:<9.2f}% | "
            f"{val_loss:<10.4f} | "
            f"{val_acc * 100:<9.2f}% | "
            f"{val_macro_f1 * 100:<12.2f}% | "
            f"{epoch_duration:.1f}s"
        )

        # Simpan checkpoint jika skor Macro F1 validasi meningkat
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": args.model_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_acc": val_acc,
                    "class_names": CLASS_NAMES,
                },
                CHECKPOINT_PATH,
            )
            print(f"   [+] Model terbaik disimpan ke: {CHECKPOINT_PATH.name} (Val Macro F1: {val_macro_f1 * 100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[!] Early stopping dipicu pada epoch {epoch} (tidak ada peningkatan selama {PATIENCE} epoch).")
                break

    total_time = time.time() - start_time
    print(f"\nPelatihan selesai dalam {total_time / 60:.2f} menit.")

    # 6. Evaluasi Final pada Data Test Independen
    print("\n" + "=" * 70)
    print("  EVALUASI MODEL TERBAIK PADA TEST SET")
    print("=" * 70)

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint terbaik (Epoch {checkpoint['epoch']}) berhasil dimuat.")

    test_loss, test_acc, test_macro_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, use_bf16=args.bf16
    )

    print(f"\nHasil Evaluasi Test Set:")
    print(f"  Test Loss       : {test_loss:.4f}")
    print(f"  Overall Accuracy: {test_acc * 100:.2f}%")
    print(f"  Macro F1-Score  : {test_macro_f1 * 100:.2f}%")

    print("\nClassification Report (Per-Class Metrics):")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Simpan metrik ke JSON
    metrics_summary = {
        "model_type": args.model_type,
        "best_epoch": checkpoint.get("epoch", args.epochs),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "confusion_matrix": cm,
        "class_names": CLASS_NAMES,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrik lengkap disimpan ke: {METRICS_PATH}")


if __name__ == "__main__":
    main()
