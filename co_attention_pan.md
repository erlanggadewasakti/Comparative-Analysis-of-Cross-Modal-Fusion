Anda adalah seorang AI & Machine Learning Engineer ahli di bidang Multimodal Deep Learning. Saya sedang mengerjakan penelitian Analisis Sentimen Multimodal (Teks + Gambar).

Saya sudah memiliki starter code (pandas) yang menghasilkan dataframe bernama `df` dengan kolom: `id`, `text`, `image_path`, dan `label` (0: negative, 1: neutral, 2: positive).

Tugas Anda: Buatkan script PyTorch lanjutan dari starter code saya untuk membuat model **Co-Attention Fusion (Alternating/Symmetric)**, melakukan training, dan testing.

Harap penuhi spesifikasi arsitektur dan pipeline berikut secara konsisten:

1. **Dataset & DataLoader:** Custom PyTorch `Dataset`. Tokenizer `roberta-base` (max_length=128), gambar resize 224x224. Split data: 80% Train, 10% Val, 10% Test.
2. **Unimodal Encoders:** - Teks: `roberta-base` (_sequence of tokens_).
   - Gambar: `vit_base_patch16_224` (_sequence of patches_).
   - Proyeksikan dimensi teks dan gambar ke dimensi ruang bersama (_joint space dimension_, misal $d=512$).
3. **Fusion Mechanism (KUNCI - Co-Attention):** - Buat class `CoAttentionFusionModel`.
   - Hitung matriks atensi bersama (Affinity Matrix) antara urutan teks dan patch gambar (misal menggunakan perkalian matriks $C = \tanh(X_t \cdot W \cdot X_v^T)$).
   - Gunakan matriks ini untuk memproyeksikan fitur teks ke ruang gambar, dan fitur gambar ke ruang teks secara bersamaan.
   - Gabungkan (concat) representasi akhir teks dan gambar yang sudah saling teratensi, lalu klasifikasikan (3 kelas).
4. **Training & Validation Loop:** Gunakan AdamW, CrossEntropyLoss. Training loop dengan validasi. Simpan model terbaik (Best Val Loss).
5. **Testing & Metrics:** Cetak `classification_report` di Test Set, tekan pada Accuracy dan **Macro F1-Score**.

Berikan kodenya secara lengkap tanpa mengulang starter code. Berikan komentar jelas pada bagian perhitungan Affinity Matrix di Co-Attention.
