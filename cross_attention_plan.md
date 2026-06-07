Anda adalah seorang AI & Machine Learning Engineer ahli di bidang Multimodal Deep Learning. Saya sedang mengerjakan penelitian Analisis Sentimen Multimodal (Teks + Gambar).

Saya sudah memiliki starter code (pandas) yang menghasilkan dataframe bernama `df` dengan kolom: `id`, `text`, `image_path`, dan `label` (0: negative, 1: neutral, 2: positive).

Tugas Anda: Buatkan script PyTorch lanjutan dari starter code saya untuk membuat model **Standard Cross-Attention Fusion**, melakukan training, dan testing.

Harap penuhi spesifikasi arsitektur dan pipeline berikut secara konsisten:

1. **Dataset & DataLoader:** Buat Custom PyTorch `Dataset`. Gunakan `roberta-base` tokenizer untuk teks (max_length=128) dan torchvision/timm transforms (resize 224x224, normalize) untuk gambar. Split data: 80% Train, 10% Val, 10% Test.
2. **Unimodal Encoders (Perlu Sequence Features):** - Teks: `roberta-base` (ambil _sequence of tokens_ output, bukan cuma CLS).
   - Gambar: Gunakan Vision Transformer (`vit_base_patch16_224`) untuk mendapatkan _sequence of patches_ output.
   - Samakan dimensi fitur teks dan gambar menggunakan layer proyeksi linier (misal ke dimensi 512).
3. **Fusion Mechanism (KUNCI):** - Buat class `CrossAttentionFusionModel`.
   - Gunakan `nn.MultiheadAttention` dari PyTorch.
   - Jadikan fitur Teks sebagai Query (Q), dan fitur Gambar sebagai Key (K) & Value (V) -- (Text-guided Visual Attention).
   - Lakukan pooling (misal mean pooling) pada hasil cross-attention, lalu masukkan ke classifier 3 kelas.
4. **Training & Validation Loop:** Gunakan AdamW optimizer, CrossEntropyLoss. Buat loop training standar dengan validasi di setiap epoch. Simpan model terbaik berdasarkan Validation Loss.
5. **Testing & Metrics:** Evaluasi di Test Set. Tampilkan Accuracy dan **Macro F1-Score** menggunakan `classification_report` dari scikit-learn.

Berikan kodenya secara lengkap (asumsikan `df` sudah ada). Pastikan arsitektur training loop sama persis dengan standar PyTorch.
