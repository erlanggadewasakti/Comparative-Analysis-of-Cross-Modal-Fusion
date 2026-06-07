Anda adalah seorang AI & Machine Learning Engineer ahli di bidang Multimodal Deep Learning. Saya sedang mengerjakan penelitian Analisis Sentimen Multimodal (Teks + Gambar).

Saya sudah memiliki starter code (pandas) yang menghasilkan dataframe bernama `df` dengan kolom: `id`, `text`, `image_path`, dan `label` (0: negative, 1: neutral, 2: positive).

Tugas Anda: Buatkan script PyTorch lanjutan dari starter code saya untuk membuat model **Hierarchical Cross-Attention Fusion**, melakukan training, dan testing.

Harap penuhi spesifikasi arsitektur dan pipeline berikut secara konsisten:

1. **Dataset & DataLoader:** Custom PyTorch `Dataset`. Tokenizer `roberta-base` (max_length=128), ViT image transform 224x224. Split: 80% Train, 10% Val, 10% Test.
2. **Unimodal Encoders:** - Teks: `roberta-base` (keluarkan _sequence output_ DAN _pooled output/CLS_).
   - Gambar: `vit_base_patch16_224` (keluarkan _patch sequence_ DAN _global CLS token_).
   - Samakan dimensi menggunakan linear layer.
3. **Fusion Mechanism (KUNCI - Hierarchical):** - Buat class `HierarchicalFusionModel`.
   - **Level 1 (Local):** Lakukan Cross-Attention antara _sequence tokens_ teks dan _patch sequence_ gambar (mencari kecocokan kata vs area gambar). Lakukan pooling pada output ini.
   - **Level 2 (Global):** Lakukan fusi (bisa dengan attention atau gating) antara representasi global (_pooled CLS teks_ dan _global CLS gambar_) dengan output dari Level 1.
   - Masukkan hasil hierarki ini ke layer klasifikasi 3 kelas.
4. **Training & Validation Loop:** AdamW, CrossEntropyLoss. Loop standar, simpan model dengan Best Val Loss.
5. **Testing & Metrics:** Evaluasi menggunakan Test Set. Tampilkan hasil Accuracy dan **Macro F1-Score**.

Tuliskan kode PyTorch yang rapi dan siap eksekusi melanjutkan dari `df`. Pastikan algoritma training loop dan DataLoader persis sama bentuknya dengan model klasifikasi standar agar mudah saya bandingkan apple-to-apple dengan model lain.
