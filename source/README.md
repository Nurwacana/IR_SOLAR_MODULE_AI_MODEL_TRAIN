# Panduan Pipeline Klasifikasi 12 Kelas Anomali Modul Surya

Pipeline ini dirancang untuk mendeteksi dan mengklasifikasikan 12 tipe kondisi modul fotovoltaik pada dataset **InfraredSolarModules** menggunakan arsitektur deep learning yang tangguh (ResNet-50 / EfficientNet) dengan penanganan ketidakseimbangan kelas (*imbalance handling*).

---

## 1. Struktur Modul di Folder `source/`

- [`config.py`](file:///D:/00_Kuliah/Kecerdasan%20Buatan/Model%20panel%20surya/source/config.py): Konfigurasi path data, hyperparameter, daftar 12 kelas, ukuran input citra, rasio pembagian data, dan device.
- [`dataset.py`](file:///D:/00_Kuliah/Kecerdasan%20Buatan/Model%20panel%20surya/source/dataset.py): Stratified dataset splitting (Train 70%, Val 15%, Test 15%), data loader, augmentasi spasial termal, dan kalkulator *Class Weights*.
- [`model.py`](file:///D:/00_Kuliah/Kecerdasan%20Buatan/Model%20panel%20surya/source/model.py): Arsitektur deep model (ResNet-50 dengan Squeeze-and-Excitation Attention + Dual Pooling Head, atau EfficientNet-B0).
- [`train.py`](file:///D:/00_Kuliah/Kecerdasan%20Buatan/Model%20panel%20surya/source/train.py): Script pelatihan dengan Weighted Cross-Entropy Loss, Cosine Annealing LR Scheduler, Early Stopping berbasis Macro F1-Score, dan evaluasi matriks lengkap.
- [`inference.py`](file:///D:/00_Kuliah/Kecerdasan%20Buatan/Model%20panel%20surya/source/inference.py): Script inferensi/prediksi mandiri untuk citra tunggal, batch direktori, atau sampel acak dengan interpretasi fisik anomali.

---

## 2. Pilihan Arsitektur Model yang Tersedia

Anda dapat memilih berbagai tingkat bobot arsitektur model sesuai kebutuhan komputasi:

| Model Type | Perkiraan Parameter | Karakteristik & Rekomendasi Penggunaan | Kecepatan Training |
|:---|:---:|:---|:---:|
| `compact_cnn` | **~450 Ribu** | **Ultra-Ringan:** Dibuat khusus dari nol (*scratch*), sangat cepat di CPU laptop, ukuran file model < 2 MB. | ⚡⚡⚡⚡⚡ Tercepat |
| `mobilenetv3` | **~2.5 Juta** | **Sangat Ringan:** Arsitektur MobileNetV3-Small, hemat memori, ideal untuk drone/edge deployment. | ⚡⚡⚡⚡ Sangat Cepat |
| `efficientnet` | **~4.0 Juta** | **Efisien & Akurat:** EfficientNet-B0 dengan compound scaling dan Squeeze-and-Excitation (SE) block. | ⚡⚡⚡ Cepat |
| `resnet18` | **~11.2 Juta** | **Paling Seimbang (Direkomendasikan):** Residual network klasik, sangat stabil, acuan utama literatur. | ⚡⚡ Sedang |
| `resnet50` | **~25.5 Juta** | **Dalam/Berat:** Fitur spasial mendalam untuk klasifikasi anomali kompleks. | ⚡ Paling Lambat |

---

## 3. Cara Menjalankan Pelatihan (Training)

Buka terminal PowerShell dan navigasikan ke folder `source/`:

```powershell
cd "D:\00_Kuliah\Kecerdasan Buatan\Model panel surya\source"
```

### A. Rekomendasi Model Ringan & Cepat (ResNet-18) — *Paling Seimbang*:
```powershell
# Di GPU Intel Arc (XPU)
python train.py --model_type resnet18 --epochs 25 --batch_size 32 --device xpu --use_ipex --bf16

# Di CPU Intel Core Ultra
python train.py --model_type resnet18 --epochs 25 --batch_size 32 --device cpu --bf16
```

### B. Model Ultra-Ringan (Compact CNN) — *Eksekusi Tercepat*:
```powershell
# Di CPU Intel Core Ultra (hanya butuh hitungan detik per epoch)
python train.py --model_type compact_cnn --epochs 25 --batch_size 32 --device cpu
```

### C. Model MobileNetV3-Small:
```powershell
python train.py --model_type mobilenetv3 --epochs 25 --batch_size 32 --device xpu --use_ipex --bf16
```

### D. Model ResNet-50 (Mendalam / Heavy):
```powershell
python train.py --model_type resnet50 --epochs 25 --batch_size 32 --device xpu --use_ipex --bf16
```

> **Catatan:**
> - Selama pelatihan, model terbaik akan otomatis disimpan ke: `source/outputs/best_solar_model.pth`.
> - Evaluasi test set mandiri (Classification Report, Confusion Matrix, Macro F1) akan otomatis disimpan ke: `source/outputs/evaluation_metrics.json`.

---

## 3. Cara Menjalankan Inferensi (Prediksi)

### A. Uji Coba Cepat pada Sampel Dataset Tertentu:
```powershell
python inference.py --sample_id 13357
```

### B. Prediksi File Citra Tunggal:
```powershell
python inference.py --image "D:\00_Kuliah\Kecerdasan Buatan\Model panel surya\infraredsolarmodules\2020-02-14_InfraredSolarModules\InfraredSolarModules\images\100.jpg"
```

### C. Prediksi Seluruh Citra dalam Sebuah Folder (Batch):
```powershell
python inference.py --dir "D:\00_Kuliah\Kecerdasan Buatan\Model panel surya\infraredsolarmodules\2020-02-14_InfraredSolarModules\InfraredSolarModules\images"
```

---

## 4. Penggunaan Langsung di Python Script Lain

```python
from pathlib import Path
from inference import SolarAnomalyPredictor

# Inisialisasi predictor
predictor = SolarAnomalyPredictor(
    checkpoint_path=Path("outputs/best_solar_model.pth"),
    model_type="resnet50"
)

# Prediksi
hasil = predictor.predict("path/ke/citra.jpg")
print(f"Prediksi: {hasil['predicted_class']} ({hasil['confidence_percentage']})")
print(f"Status  : {hasil['status']}")
print(f"Tindakan: {hasil['rekomendasi']}")
```

---

## 5. Sumber Dataset

* **Dataset Name:** InfraredSolarModules
* **Organisasi / Penulis:** Raptor Maps Inc. (ICLR 2020 AI for Earth Sciences Workshop)
* **URL Repositori Dataset:** https://github.com/RaptorMaps/InfraredSolarModules
* **Makalah Resmi:** https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf
