# IR_SOLAR_MODULE_AI_MODEL_TRAIN

Pipeline *Deep Learning* canggih untuk deteksi dan klasifikasi 12 kelas anomali/kerusakan pada modul panel surya fotovoltaik (PV) berbasis citra termografi inframerah (*Infrared Thermography*). 

Proyek ini dibangun di atas dataset publik **InfraredSolarModules** (Raptor Maps / ICLR 2020 AI for Earth Sciences Workshop) dan dioptimalkan dengan akselerasi perangkat keras **Intel Extension for PyTorch (IPEX)** pada CPU Intel Core Ultra maupun kartu grafis **Intel Arc Graphics (XPU)**.

---

## 🚀 Fitur Utama

- **Klasifikasi 12 Kelas Anomali:**
  - 1 Kelas Nominal: `No-Anomaly`
  - 11 Kelas Kerusakan: `Cell`, `Cell-Multi`, `Cracking`, `Diode`, `Diode-Multi`, `Hot-Spot`, `Hot-Spot-Multi`, `Offline-Module`, `Shadowing`, `Soiling`, `Vegetation`.
- **Penanganan Ketidakseimbangan Ekstrem (Class Imbalance):**
  - Menggunakan *Stratified K-Fold / Split* (70% Train, 15% Val, 15% Test).
  - *Square-Root Dampened Class Weights* pada *Cross-Entropy Loss* untuk melindungi kelas minoritas tanpa mengorbankan presisi kelas nominal.
  - *Label Smoothing Regularization (0.05)* untuk membedakan anomali visual yang berdekatan.
- **Pilihan 5 Arsitektur Model Fleksibel:**
  1. `compact_cnn` (~626K params) — *Ultra-lightweight custom CNN dari scratch*.
  2. `mobilenetv3` (~1.07M params) — *MobileNetV3-Small untuk drone/edge deployment*.
  3. `efficientnet` (~4.33M params) — *EfficientNet-B0 dengan SE-Block*.
  4. `resnet18` (~11.47M params) — *Paling seimbang & stabil (acuan benchmark)*.
  5. `resnet50` (~26.26M params) — *Deep feature extractor*.
- **Akselerasi Intel oneAPI:**
  - Mendukung backend GPU `torch.xpu` (Intel Arc).
  - Mendukung CPU Multi-threading Intel oneDNN (MKL-DNN).
  - Akselerasi format komputasi modern **Bfloat16 (AMP)**.

---

## 📁 Struktur Repositori

```
├── .gitignore
├── README.md
├── requirements.txt
├── source/
│   ├── config.py           # Konfigurasi path, parameter, dan deteksi hardware
│   ├── dataset.py          # PyTorch Dataset loader, augmentasi spasial, class weights
│   ├── model.py            # Definisi 5 model deep learning + attention head
│   ├── train.py            # Loop pelatihan, scheduler, early stopping Macro F1
│   ├── inference.py        # Mesin prediksi mandiri (file/batch) + diagnosis fisik
│   └── README.md           # Panduan teknis eksekusi modul source
└── infraredsolarmodules/   # Dataset citra termal modul surya
```

---

## 🛠️ Instalasi & Persiapan Lingkungan

1. **Clone Repositori:**
   ```powershell
   git clone https://github.com/Nurwacana/IR_SOLAR_MODULE_AI_MODEL_TRAIN.git
   cd IR_SOLAR_MODULE_AI_MODEL_TRAIN
   ```

2. **Buat dan Aktifkan Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Pasang Dependensi:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 📊 Sumber & Unduhan Dataset (Dataset Source)

Dataset yang digunakan dalam repositori ini adalah **InfraredSolarModules**, yaitu kumpulan data citra inframerah/termografi modul fotovoltaik beranotasi yang dirilis oleh **Raptor Maps**:

* **Repositori Resmi Dataset:** [github.com/RaptorMaps/InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules)
* **Publikasi Ilmiah (Paper):** [Infrared Solar Module Dataset for Anomaly Detection (ICLR 2020 Workshop)](https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf)
* **Tautan Unduh Langsung Arsip Citra:** [2020-02-14_InfraredSolarModules.zip](https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip) (~15.5 MB)

### Cara Memasang Dataset ke dalam Proyek Ini:

Karena repositori Git mengecualikan file citra berukuran besar via `.gitignore`, siapkan dataset dengan langkah berikut:

```powershell
# Opsi 1: Clone repositori dataset langsung ke folder infraredsolarmodules
git clone https://github.com/RaptorMaps/InfraredSolarModules.git infraredsolarmodules

# Opsi 2: Ekstrak arsip zip dataset
# Pastikan folder ekstraksi tersusun sesuai struktur:
# infraredsolarmodules/2020-02-14_InfraredSolarModules/InfraredSolarModules/
#   ├── images/ (berisi 0.jpg s.d. 19999.jpg)
#   └── module_metadata.json
```

---

## ⚡ Cara Menjalankan Pelatihan (Training)

Masuk ke folder `source/`:
```powershell
cd source
```

### 1. Pelatihan di GPU Intel Arc (XPU) — *Paling Cepat*:
```powershell
python train.py --model_type resnet18 --epochs 30 --batch_size 64 --device xpu --use_ipex --bf16
```

### 2. Pelatihan di CPU Intel Core Ultra:
```powershell
python train.py --model_type resnet18 --epochs 30 --batch_size 64 --device cpu --bf16
```

### 3. Model Ultra-Ringan (Compact CNN):
```powershell
python train.py --model_type compact_cnn --epochs 25 --batch_size 32 --device cpu
```

---

## 🔍 Inferensi & Prediksi

```powershell
# Prediksi sampel acak dari dataset
python inference.py --sample_id 13357

# Prediksi citra tunggal
python inference.py --image "path/ke/citra_termal.jpg"
```

---

## 📄 Lisensi & Referensi (Citations)

Dataset bersumber dari penelitian **Raptor Maps Inc.** yang dipublikasikan pada *International Conference on Learning Representations (ICLR) 2020 Workshop on AI for Earth Sciences*. Repositori ini didistribusikan di bawah lisensi MIT.

Jika menggunakan dataset ini dalam penelitian akademis, silakan merujuk pada sitasi berikut:

```bibtex
@inproceedings{millendorf2020infrared,
  title={Infrared Solar Module Dataset for Anomaly Detection},
  author={Millendorf, Matthew and Obropta, Edward and Vadhavkar, Nikhil},
  booktitle={ICLR 2020 Workshop on AI for Earth Sciences},
  year={2020},
  url={https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf}
}
```
