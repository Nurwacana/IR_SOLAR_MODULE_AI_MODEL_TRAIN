"""
Script Inferensi dan Prediksi Anomali Modul Surya (12 Kelas)
Dapat digunakan untuk inferensi file tunggal, direktori gambar, atau sampel acak.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DEVICE,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    IMAGES_DIR,
    METADATA_PATH,
    NUM_CLASSES,
)
from model import build_model

# Kamus Penjelasan dan Tindakan Perbaikan Anomali
ANOMALY_INFO = {
    "No-Anomaly": {
        "status": "NORMAL",
        "deskripsi": "Modul beroperasi nominal. Distribusi suhu seragam tanpa titik panas abnormal.",
        "rekomendasi": "Tidak ada tindakan yang diperlukan.",
    },
    "Cell": {
        "status": "ANOMALI",
        "deskripsi": "Terdeteksi Hot Spot lokal berbentuk bujur sangkar pada satu sel surya.",
        "rekomendasi": "Periksa adanya micro-crack fisik atau degradasi sel. Pantau risiko kebakaran lokal.",
    },
    "Cell-Multi": {
        "status": "ANOMALI",
        "deskripsi": "Terdeteksi Hot Spot pada beberapa sel surya dalam satu modul.",
        "rekomendasi": "Inspeksi koneksi interkoneksi sel. Segera jadwalkan penggantian jika suhu terus meningkat.",
    },
    "Cracking": {
        "status": "ANOMALI",
        "deskripsi": "Retakan fisik pada permukaan kaca atau lapisan silikon modul surya.",
        "rekomendasi": "Periksa keutuhan struktural panel; cegah penetrasi air hujan yang memicu arus pendek.",
    },
    "Diode": {
        "status": "KRITIS",
        "deskripsi": "Bypass diode aktif/rusak, menyebabkan pemanasan memanjang pada sepertiga modul (~1/3).",
        "rekomendasi": "Ganti bypass diode pada junction box untuk memulihkan sepertiga daya modul yang hilang.",
    },
    "Diode-Multi": {
        "status": "KRITIS",
        "deskripsi": "Multi bypass diode aktif, menyebabkan dua pertiga modul (~2/3) memanas tinggi.",
        "rekomendasi": "Prioritas tinggi: ganti diode junction box atau modul untuk mencegah kerusakan permanen.",
    },
    "Hot-Spot": {
        "status": "ANOMALI",
        "deskripsi": "Titik panas terfokus pada modul tipe thin-film.",
        "rekomendasi": "Periksa delaminasi lapisan tipis dan ukur penurunan efisiensi arus string.",
    },
    "Hot-Spot-Multi": {
        "status": "KRITIS",
        "deskripsi": "Banyak titik panas ekstrem pada modul tipe thin-film (potensi suhu tinggi).",
        "rekomendasi": "Isolasi modul yang terdampak untuk mencegah hotspot merambat ke modul lain.",
    },
    "Offline-Module": {
        "status": "KRITIS",
        "deskripsi": "Seluruh modul surya memanas merata (modul dalam kondisi open-circuit / mati).",
        "rekomendasi": "Periksa kabel konektor MC4, kabel putus, atau inverter string yang terputus.",
    },
    "Shadowing": {
        "status": "PERINGATAN",
        "deskripsi": "Sinar matahari terhalang oleh bayangan pohon, tiang, atau deretan modul sekitar.",
        "rekomendasi": "Evaluasi pemangkasan pohon atau susunan tilting panel untuk meminimalkan bayangan.",
    },
    "Soiling": {
        "status": "PERINGATAN",
        "deskripsi": "Permukaan modul tertutup debu, kotoran burung, atau endapan pasir tebal.",
        "rekomendasi": "Jadwalkan pembersihan (washing) modul surya menggunakan air demineralisasi.",
    },
    "Vegetation": {
        "status": "PERINGATAN",
        "deskripsi": "Tumbuhan semak atau rumput liar menutupi sebagian modul.",
        "rekomendasi": "Lakukan pemotongan vegetasi di area sekitar array panel surya.",
    },
}


class SolarAnomalyPredictor:
    """Kelas inferensi untuk prediksi anomali modul surya."""

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        model_type: str = "resnet50",
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.model = build_model(model_type=model_type, num_classes=NUM_CLASSES, pretrained=False)

        if not Path(checkpoint_path).exists():
            print(f"[Peringatan] Checkpoint '{checkpoint_path}' tidak ditemukan. Model menggunakan bobot acak.")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Info] Berhasil memuat model dari: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_input: Union[str, Path, Image.Image], top_k: int = 3) -> Dict:
        """
        Melakukan prediksi terhadap satu citra.

        Args:
            image_input: Path ke file gambar atau objek PIL Image
            top_k: Jumlah kandidat prediksi teratas yang dikembalikan

        Returns:
            Dictionary berisi kelas prediksi, confidence, top-k ranking, serta diagnosis.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError("image_input harus berupa path string/Path atau PIL Image.")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "rank": i + 1,
                "class": IDX_TO_CLASS[idx],
                "probability": float(probs[idx]),
                "percentage": f"{probs[idx] * 100:.2f}%",
            }
            for i, idx in enumerate(top_indices)
        ]

        best_idx = top_indices[0]
        predicted_class = IDX_TO_CLASS[best_idx]
        confidence = float(probs[best_idx])
        info = ANOMALY_INFO.get(predicted_class, {})

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "status": info.get("status", "UNKNOWN"),
            "deskripsi": info.get("deskripsi", "-"),
            "rekomendasi": info.get("rekomendasi", "-"),
            "top_predictions": top_predictions,
        }


def main():
    parser = argparse.ArgumentParser(description="Inferensi Model Anomali Panel Surya (12 Kelas)")
    parser.add_argument("--image", type=str, help="Path ke satu file gambar .jpg")
    parser.add_argument("--dir", type=str, help="Path ke direktori gambar untuk inferensi batch")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Path model checkpoint .pth")
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        choices=["compact_cnn", "mobilenetv3", "efficientnet", "resnet18", "resnet50"],
        help="Pilihan arsitektur model",
    )
    parser.add_argument("--sample_id", type=str, help="ID gambar dari dataset untuk pengujian langsung (contoh: 13357)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "xpu", "cuda"], help="Pilihan perangkat inferensi")
    args = parser.parse_args()

    # Tentukan perangkat inferensi
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "xpu":
        device = torch.device("xpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = DEVICE

    predictor = SolarAnomalyPredictor(
        checkpoint_path=Path(args.checkpoint),
        model_type=args.model_type,
        device=device,
    )

    # 1. Kasus input satu file gambar
    if args.image:
        result = predictor.predict(args.image)
        print("\n" + "=" * 60)
        print(f"HASIL PREDIKSI: {args.image}")
        print("=" * 60)
        print(f"Prediksi Kelas : {result['predicted_class']} ({result['confidence_percentage']})")
        print(f"Status Panel   : {result['status']}")
        print(f"Deskripsi      : {result['deskripsi']}")
        print(f"Rekomendasi    : {result['rekomendasi']}")
        print("\nTop-3 Prediksi:")
        for top in result["top_predictions"]:
            print(f"  #{top['rank']}: {top['class']:<16} ({top['percentage']})")

    # 2. Kasus pengujian sample ID dataset
    elif args.sample_id:
        sample_path = IMAGES_DIR / f"{args.sample_id}.jpg"
        if not sample_path.exists():
            print(f"Gambar dengan ID {args.sample_id} tidak ditemukan di {IMAGES_DIR}")
            return

        # Cek ground truth jika ada di metadata
        ground_truth = "Unknown"
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if args.sample_id in meta:
                    ground_truth = meta[args.sample_id]["anomaly_class"]

        result = predictor.predict(sample_path)
        print("\n" + "=" * 60)
        print(f"PENGUJIAN SAMPEL DATASET ID: {args.sample_id}")
        print(f"Ground Truth Label: {ground_truth}")
        print("=" * 60)
        print(f"Prediksi Kelas    : {result['predicted_class']} ({result['confidence_percentage']})")
        print(f"Status            : {result['status']}")
        print(f"Deskripsi         : {result['deskripsi']}")
        print(f"Rekomendasi       : {result['rekomendasi']}")
        print("\nTop-3 Prediksi:")
        for top in result["top_predictions"]:
            print(f"  #{top['rank']}: {top['class']:<16} ({top['percentage']})")

    # 3. Kasus input direktori gambar (Batch)
    elif args.dir:
        dir_path = Path(args.dir)
        image_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        print(f"\nMemproses {len(image_files)} citra di '{dir_path}'...")

        batch_results = []
        for img_p in image_files[:50]:  # Batasi preview 50 pertama jika banyak
            res = predictor.predict(img_p)
            batch_results.append({
                "file": img_p.name,
                "predicted_class": res["predicted_class"],
                "confidence": res["confidence_percentage"],
                "status": res["status"],
            })

        print(f"{'Nama File':<15} | {'Prediksi Kelas':<16} | {'Confidence':<12} | {'Status'}")
        print("-" * 55)
        for r in batch_results:
            print(f"{r['file']:<15} | {r['predicted_class']:<16} | {r['confidence']:<12} | {r['status']}")

    else:
        print("Gunakan argumen --image <path>, --sample_id <id>, atau --dir <folder_path> untuk menjalankan prediksi.")


if __name__ == "__main__":
    main()
