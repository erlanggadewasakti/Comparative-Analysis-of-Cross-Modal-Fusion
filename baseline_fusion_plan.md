Anda adalah seorang AI & Machine Learning Engineer ahli di bidang Multimodal Deep Learning. Saya sedang mengerjakan penelitian Analisis Sentimen Multimodal (Teks + Gambar).

Saya sudah memiliki starter code (pandas) yang menghasilkan dataframe bernama `df` dengan kolom: `id`, `text`, `image_path`, dan `label` (0: negative, 1: neutral, 2: positive).

Tugas Anda: Buatkan script PyTorch lanjutan dari starter code saya untuk membuat model **Baseline: Self-Attention + Concatenation (Late Fusion)**, melakukan training, dan testing.

Harap penuhi spesifikasi arsitektur dan pipeline berikut secara konsisten:

1. **Dataset & DataLoader:** Buat Custom PyTorch `Dataset`. Gunakan `roberta-base` tokenizer untuk teks (max_length=128) dan torchvision transforms standar (resize 224x224, normalize) untuk gambar. Split data menjadi 80% Train, 10% Val, 10% Test.
2. **Unimodal Encoders:** - Teks: Gunakan `roberta-base` (ambil token `[CLS]` atau pooled output).
   - Gambar: Gunakan `resnet50` atau `vit_base_patch16_224` (ambil representasi global/pooled output).
   - Freeze layer bawah pada encoder jika perlu agar tidak Out of Memory (OOM).
3. **Fusion Mechanism (KUNCI):** - Buat class `BaselineFusionModel`.
   - Gabungkan (Concatenate) vektor fitur teks dan vektor fitur gambar secara langsung.
   - Masukkan hasil gabungan ke layer MLP/Linear classifier untuk output 3 kelas.
4. **Training & Validation Loop:** Gunakan AdamW optimizer, CrossEntropyLoss. Buat loop training standar dengan validasi di setiap epoch. Simpan model terbaik berdasarkan Validation Loss.
5. **Testing & Metrics:** Setelah training selesai, evaluasi model pada Test Set. Tampilkan Classification Report (Accuracy, Precision, Recall, dan khusus tekan pada **Macro F1-Score**).

Berikan kodenya secara lengkap (hanya bagian PyTorch saja, asumsikan `df` sudah ada di memori). Berikan komentar penjelas di bagian Fusion.
