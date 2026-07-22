
Anda adalah **AI Lead Code Engineer** yang ahli dalam bidang **Multimodal Sentiment Analysis (MSA)**, **Deep Learning**, **Evidential Deep Learning (EDL)**, dan **Teori Probabilitas Dempster-Shafer (DST)**.

Tugas Anda adalah menulis kode implementasi **PyTorch** yang bersih, optimal, terkomentari dengan baik, dan siap dijalankan (*production-ready*) untuk model Tesis S2 saya yang berjudul: **"Adaptive Evidential Fusion (ADEF) with Co-Attention"**.

Arsitektur model ini secara ketat dirancang berdasarkan rancangan resmi proposal tesis saya (Erlangga Dewa Sakti, Telkom University) di bawah bimbingan **Prof. Dr. ADIWIJAYA**. Model ini menggabungkan ekstraksi korelasi silang teks-visual menggunakan **Bidirectional Co-Attention tingkat sekuens (token–patch)** dengan kerangka kuantifikasi ketidakpastian berbasis **Evidential Deep Learning (EDL)**.

> **Revisi v2** — Dokumen ini adalah revisi dari rancangan v1 berdasarkan hasil eksperimen awal. Model v1 mengalami *class collapse* total pada kelas **Netral** (Precision = Recall = F1 = 0.00, Macro-F1 = 0.43 — terendah di antara 5 model pembanding). Analisis akar masalah dan perbaikan yang dilakukan dirangkum pada **Lampiran A (Catatan Revisi)**. Perubahan utama v2: (1) Co-Attention dinaikkan ke tingkat sekuens (token–patch), (2) kompensasi ketimpangan kelas, (3) seluruh hyperparameter (termasuk τ) menjadi terkonfigurasi, (4) seleksi model berbasis Macro-F1, dan (5) supervisi langsung pada opini hasil fusi.

---

### I. SPESIFIKASI INPUT & OUTPUT DATA
1. **Input Teks ($T$):** Tokenized text IDs dengan ukuran `[batch_size, seq_len]` dan Attention Mask `[batch_size, seq_len]`. Panjang sekuens maksimal ($L_t$) adalah **150**.
2. **Input Gambar ($V$):** Tensor gambar RGB dengan ukuran `[batch_size, 3, 224, 224]`.
3. **Output Prediksi ($\hat{y}$):** Label sentimen mayoritas dari **3 kelas** (0: Negatif, 1: Netral, 2: Positif).
4. **Distribusi Kelas (MVSA-Single setelah filtering):** sekitar **59% Positif / 30% Negatif / 10% Netral**. Ketimpangan ini **wajib** dikompensasi (lihat §III.4) — tanpa kompensasi, kelas Netral mengalami *collapse*.
5. **Pembagian Data:** *Stratified split* 70% train / 15% validasi / 15% test dengan seed tetap, identik dengan 4 notebook baseline (early/late/cross/co-attention) demi keadilan komparasi.

---

### II. STRUKTUR ARSITEKTUR MODEL (ADEFCoAttnNet v2)

Terapkan model PyTorch modular yang terdiri dari komponen-komponen berikut:

#### 1. Unimodal Feature Extraction (Sequence-Level Feature Extraction)
Berbeda dengan v1 yang hanya mengambil vektor ter-*pooling*, v2 mempertahankan **representasi tingkat sekuens** dari kedua encoder agar Co-Attention dapat melakukan penyelarasan halus (*fine-grained alignment*). Kedua backbone **dibekukan (*frozen*)** agar komparasi dengan 4 model baseline tetap adil.

*   **Text Encoder:** Gunakan Pre-trained Language Model `RoBERTa-base`. Ambil **seluruh hidden state** dari lapisan terakhir, lalu proyeksikan setiap token dengan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi laten bersama $d = 512$:
    $$\mathbf{H}_t = f_{\text{RoBERTa}}(T) \in \mathbb{R}^{L_t \times d}, \qquad L_t = 150$$
    Vektor terpooling untuk kepala unimodal diperoleh dengan *masked mean-pooling*:
    $$\mathbf{h}_t = \frac{\sum_{l=1}^{L_t} m_l \mathbf{H}_{t,l}}{\sum_{l=1}^{L_t} m_l} \in \mathbb{R}^{d}$$
    dengan $m_l$ adalah attention mask.
*   **Image Encoder:** Gunakan `DenseNet121` pre-trained. Ambil **feature map spasial** sebelum Global Average Pooling, $[\,1024 \times 7 \times 7\,]$, lalu ratakan menjadi $N_v = 49$ *patch* dan proyeksikan dengan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi $d = 512$:
    $$\mathbf{H}_v = f_{\text{DenseNet}}(V) \in \mathbb{R}^{N_v \times d}, \qquad N_v = 49$$
    Vektor terpooling untuk kepala unimodal:
    $$\mathbf{h}_v = \frac{1}{N_v} \sum_{n=1}^{N_v} \mathbf{H}_{v,n} \in \mathbb{R}^{d}$$

#### 2. Bidirectional Co-Attention Module (Sequence-Level)
Modul ini menangkap interaksi dan penyelarasan semantik halus dua arah antara **kata-kata teks** dan **area visual** sebelum estimasi ketidakpastian.

*   **Matriks Afinitas ($S$):** Hitung skor afinitas silang menggunakan perkalian dot-product terproyeksi ($\mathbf{W} \in \mathbb{R}^{d \times d}$):
    $$S = \frac{\mathbf{H}_t \, \mathbf{W} \, \mathbf{H}_v^T}{\sqrt{d}} \in \mathbb{R}^{L_t \times N_v}$$
*   **Atensi Dua Arah:** Terapkan softmax pada kedua orientasi. Token padding teks ditutup (*masked*, $-\infty$) sebelum softmax arah visual$\to$teks:
    $$A_{tv} = \text{Softmax}_{\text{patch}}(S) \in \mathbb{R}^{L_t \times N_v}, \qquad A_{vt} = \text{Softmax}_{\text{token}}(S^T) \in \mathbb{R}^{N_v \times L_t}$$
*   **Representasi Teratensi:** Setiap kata menghadirkan ringkasan visual teratensi (*text-guided visual*); setiap patch menghadirkan ringkasan tekstual teratensi (*visual-guided text*):
    $$\mathbf{v}_{\text{att}} = \text{MaskedMeanPool}(A_{tv} \, \mathbf{H}_v) \in \mathbb{R}^{d}, \qquad \mathbf{t}_{\text{att}} = \text{MeanPool}(A_{vt} \, \mathbf{H}_t) \in \mathbb{R}^{d}$$
*   **Representasi Gabungan ($\mathbf{h}_c$):** Gabungkan kedua ringkasan lalu proyeksikan (Linear + ReLU + LayerNorm + Dropout):
    $$\mathbf{h}_c = \text{MLP}\big(\text{Concat}(\mathbf{t}_{\text{att}}, \mathbf{v}_{\text{att}})\big) \in \mathbb{R}^{d}$$
    $\mathbf{h}_c$ bertindak sebagai jalur ketiga yang merepresentasikan hubungan interaksi teks-gambar.

#### 3. Evidential Neural Network (ENN) Heads
Alih-alih aktivasi Softmax deterministik yang menyebabkan *overconfidence*, ketiga fitur terpooling ($\mathbf{h}_t$, $\mathbf{h}_v$, $\mathbf{h}_c$) dilewatkan secara paralel ke tiga ENN Head independen (Text ENN, Image ENN, Co-Attention ENN).
*   **Ekstraksi Bukti (*Evidence Extraction*):** Lapisan fully-connected ($d \to d/2 \to M$) dengan aktivasi non-negatif **Softplus** untuk menjamin $e \ge 0$. Untuk setiap jalur $k \in \{t, v, c\}$ dan kelas $i \in \{1, 2, 3\}$:
    $$e_{k,i} = \text{Softplus}(\mathbf{W}_k \mathbf{h}_k + \mathbf{b}_k)$$
*   **Parameter Distribusi Dirichlet ($\alpha$):**
    $$\alpha_{k,i} = e_{k,i} + 1$$
*   **Kekuatan Bukti Total ($S_k$):**
    $$S_k = \sum_{i=1}^M \alpha_{k,i}, \qquad M = 3$$

#### 4. Kuantifikasi Belief Mass & Uncertainty Mass
Berdasarkan *Subjective Logic* (SL), hitung massa keyakinan dan massa ketidakpastian global secara independen untuk masing-masing jalur:
*   **Belief Mass ($b_{k,i}$):**
    $$b_{k,i} = \frac{e_{k,i}}{S_k}$$
*   **Uncertainty Mass ($u_k$):**
    $$u_k = \frac{M}{S_k}$$
*   Aksioma Subjective Logic:
    $$\sum_{i=1}^M b_{k,i} + u_k = 1$$

#### 5. Modul Adaptive Evidential Fusion (ADEF)
Modul ini melakukan fusi adaptif dengan mengevaluasi tingkat perselisihan opini antara jalur teks murni dan gambar murni.

*   **Kalkulasi Massa Konflik ($K_{tv}$):**
    $$K_{tv} = \sum_{i=1}^M \sum_{\substack{j=1 \\ j \neq i}}^M b_{t,i} \cdot b_{v,j}$$
*   **Ambang Batas $\tau$ sebagai Hyperparameter Terkalibrasi:** $\tau$ **bukan** konstanta yang di-hardcode. Nilainya dikalibrasi terhadap skala empiris $K_{tv}$ pada data validasi (lihat §IV). Rentang pencarian yang direkomendasikan: $\tau \in [0.05,\, 0.3]$; default awal $\tau = 0.1$. (Catatan v1: $\tau = 0.5$ tidak pernah terlampaui karena rata-rata $K_{tv} \approx 0.014$, sehingga Rute B tidak pernah aktif.)

*   **Rute A: Fusi Normal (Dempster-Shafer Standard) — Jika $K_{tv} \le \tau$:**
    *   *Tahap 1 (Fusi Teks-Visual):*
        $$b_{tv,i} = \frac{1}{1 - K_{tv}} (b_{t,i} \cdot b_{v,i} + b_{t,i} \cdot u_v + b_{v,i} \cdot u_t), \qquad u_{tv} = \frac{1}{1 - K_{tv}} (u_t \cdot u_v)$$
    *   *Tahap 2 (Konsensus Akhir dengan Co-Attention):* Hitung konflik baru $K_{tvc}$ antara opini $(b_{tv}, u_{tv})$ dan jalur Co-Attention $(b_c, u_c)$:
        $$K_{tvc} = \sum_{i=1}^M \sum_{\substack{j=1 \\ j \neq i}}^M b_{tv,i} \cdot b_{c,j}$$
        $$b_{\text{fusion},i} = \frac{1}{1 - K_{tvc}} (b_{tv,i} \cdot b_{c,i} + b_{tv,i} \cdot u_c + b_{c,i} \cdot u_{tv}), \qquad u_{\text{fusion}} = \frac{1}{1 - K_{tvc}} (u_{tv} \cdot u_c)$$

*   **Rute B: Fusi Resolusi Konflik (Conflict-Aware Fusion) — Jika $K_{tv} > \tau$:**
    Aturan ortogonal Dempster standar berisiko gagal pada kontradiksi ekstrem (*Zadeh's Paradox*). Gunakan $K_{tv}$ sebagai tuas pengontrol dinamis untuk meredam opini unimodal yang bertentangan dan mengalihkan dominasi bobot kepada representasi relasional dari **Co-Attention**:
    $$b_{\text{fusion},i} = (1 - K_{tv}) \cdot \left( \frac{b_{t,i} + b_{v,i}}{2} \right) + K_{tv} \cdot b_{c,i}$$
    $$u_{\text{fusion}} = \max\Big(\epsilon,\; 1 - \sum_{i=1}^M b_{\text{fusion},i}\Big)$$

#### 6. Final Decision Making
*   **Final Expectation Probability ($p_i$):**
    $$p_i = b_{\text{fusion},i} + \frac{u_{\text{fusion}}}{M}$$
*   **Output Prediksi Sentimen Final ($\hat{y}$):**
    $$\hat{y} = \text{Argmax}(p_1, p_2, p_3)$$

---

### III. BLENDED LOSS FUNCTION (TRAINING OBJECTIVE)

Model dilatih dengan *Multi-Task Learning* yang meminimalkan gabungan kerugian evidensial tersupervisi, supervisi opini fusi, dan penalti konflik:

#### 1. Multi-Task Evidential Loss dengan Pembobotan Kelas ($L_{\text{sup}}$)
$$L_{\text{sup}} = L(\alpha_t) + L(\alpha_v) + L(\alpha_c)$$
Untuk setiap jalur $k$:
$$L(\alpha_k) = w_y \cdot L_{\text{err}}(\alpha_k) + \lambda_t L_{\text{KL}}(\alpha_k)$$
*   **Pembobotan Kelas ($w_y$) — BARU di v2:** Sampel ditimbang berbanding terbalik dengan frekuensi kelasnya agar kelas minoritas (Netral, $\approx$10% data) tidak tenggelam oleh mayoritas:
    $$w_c = \frac{N}{M \cdot n_c}, \qquad w_y = w_{c=y}$$
    ($N$ = jumlah sampel train, $n_c$ = jumlah sampel kelas $c$; menghasilkan $w \approx [1.11,\ 3.20,\ 0.56]$ untuk Negatif/Netral/Positif.)
*   **Error Term ($L_{\text{err}}$):**
    $$L_{\text{err}}(\alpha_{k}) = \sum_{j=1}^M (y_j - \hat{p}_{k,j})^2 + \frac{\hat{p}_{k,j}(1 - \hat{p}_{k,j})}{S_k + 1}, \qquad \hat{p}_{k,j} = \frac{\alpha_{k,j}}{S_k}$$
*   **Regularisasi KL-Divergence ($L_{\text{KL}}$):**
    $$L_{\text{KL}}(\alpha_k) = \log \left( \frac{\Gamma(\sum_{j=1}^M \tilde{\alpha}_{k,j})}{\Gamma(M) \prod_{j=1}^M \Gamma(\tilde{\alpha}_{k,j})} \right) + \sum_{j=1}^M (\tilde{\alpha}_{k,j} - 1) \left[ \psi(\tilde{\alpha}_{k,j}) - \psi\left(\sum_{l=1}^M \tilde{\alpha}_{k,l}\right) \right]$$
    dengan $\tilde{\alpha}_{k} = y + (1 - y) \odot \alpha_k$.
*   **Annealing Coefficient ($\lambda_t$):**
    $$\lambda_t = \min\Big(1.0,\ \frac{t}{\text{ANNEALING\_EPOCHS}}\Big), \qquad \text{ANNEALING\_EPOCHS} = 10$$

#### 2. Supervisi Opini Fusi ($L_{\text{fused}}$) — BARU di v2
Pada v1, output fusi final ($p_{\text{fusion}}$) tidak pernah muncul di dalam loss, sehingga kualitas keputusan akhir hanya dioptimalkan secara tidak langsung. Di v2, opini hasil fusi **dikonversi kembali** ke distribusi Dirichlet ekuivalen dan disupervisi langsung dengan Evidential Loss yang sama:
$$S_{\text{fusion}} = \frac{M}{\hat{u}}, \qquad \hat{u} = \text{clamp}(u_{\text{fusion}},\ u_{\min},\ 1), \qquad \alpha_{\text{fusion},i} = b_{\text{fusion},i} \cdot S_{\text{fusion}} + 1$$
$$L_{\text{fused}} = L(\alpha_{\text{fusion}})$$
dengan $u_{\min} = 0.05$ untuk stabilitas numerik. Gradien mengalir melalui aturan kombinasi Dempster ke seluruh ENN head.

#### 3. Semantic Conflict Loss ($L_{\text{con}}$)
Untuk memandu encoder mendeteksi inkongruensi emosi dan memetakan konflik semantik menjadi ketidakpastian multimodal:
$$L_{\text{con}} = d_{\text{PD}} \cdot d_{\text{CC}} = \frac{1}{2} (1 - u_t) (1 - u_v) \sum_{i=1}^M |p_{t,i} - p_{v,i}|$$

#### 4. Total Loss Keseluruhan Jaringan
$$L_{\text{overall}} = L_{\text{sup}} + \lambda_f \, L_{\text{fused}} + \gamma \, L_{\text{con}}$$
dengan $\lambda_f$ (default: 1.0) dan $\gamma$ (default: 1.0) sebagai hyperparameter balancing yang terkonfigurasi.

#### 5. Penanganan Ketimpangan Kelas & Seleksi Model — BARU di v2
1.  **Pembobotan kelas** pada $L_{\text{err}}$ (§III.1) — kompensasi utama ketimpangan 59/30/10.
2.  **Seleksi checkpoint terbaik berdasarkan Macro-F1 validasi**, *bukan* Weighted-F1. Weighted-F1 didominasi kelas mayoritas sehingga checkpoint dengan recall Netral = 0% justru terpilih sebagai "terbaik" pada v1. Macro-F1 memberi bobot setara ke setiap kelas.
3.  **Pelaporan metrik** wajib menyertakan Accuracy, Weighted-F1, **Macro-F1**, dan **F1 per kelas** agar kegagalan pada satu kelas tidak tersembunyi.

---

### IV. TABEL HYPERPARAMETER (SEMUA TERKONFIGURASI VIA `CFG`)

Seluruh hyperparameter — termasuk $\tau$ — dideklarasikan di satu kelas konfigurasi (`CFG`) sebagai *single source of truth*. Tidak boleh ada angka *hardcoded* di dalam definisi model, loss, training loop, maupun evaluasi.

| Kategori | Parameter (`CFG`) | Simbol | Nilai Default | Keterangan |
|---|---|---|---|---|
| Data | `MAX_LEN` | $L_t$ | 150 | Panjang maksimal sekuens teks |
| Data | `FILTER_CONFLICT_PAIRS` | — | `True` | Membuang pasangan teks-gambar yang kontradiktif (konsisten dgn 4 baseline; set `False` untuk ablasi) |
| Arsitektur | `D_BERT` / `D_CNN` | — | 768 / 1024 | Dimensi output RoBERTa-base / DenseNet121 |
| Arsitektur | `D_PROJ` | $d$ | 512 | Dimensi laten bersama |
| Arsitektur | `NUM_CLASSES` | $M$ | 3 | Jumlah kelas sentimen |
| Arsitektur | `DROPOUT` | — | 0.3 | Dropout pada proyeksi & ENN head |
| Optimisasi | `BATCH_SIZE` | — | 16 | Ukuran batch |
| Optimisasi | `EPOCHS` | — | 30 | Jumlah epoch |
| Optimisasi | `LR` | — | $1 \times 10^{-4}$ | Learning rate AdamW untuk seluruh parameter terlatih (backbone beku). **Catatan v1:** $2 \times 10^{-5}$ terlalu kecil untuk head yang diinisialisasi acak → *undertraining* |
| Optimisasi | `WEIGHT_DECAY` | — | $1 \times 10^{-4}$ | Weight decay AdamW |
| Optimisasi | `GRAD_CLIP` | — | 1.0 | Gradient clipping (max norm) |
| Optimisasi | `SCHED_TMAX` | — | 30 | Periode CosineAnnealingLR |
| Optimisasi | `SEED` | — | 42 | Seed reprodusibilitas |
| EDL | `ANNEALING_EPOCHS` | — | 10 | Epoch saat $\lambda_t$ mencapai 1.0 |
| EDL | `USE_CLASS_WEIGHTS` | — | `True` | Aktifkan pembobotan kelas $w_y$ pada $L_{\text{err}}$ |
| EDL | `U_MIN` | $u_{\min}$ | 0.05 | Batas bawah $u_{\text{fusion}}$ untuk rekonstruksi $\alpha_{\text{fusion}}$ |
| ADEF | **`TAU`** | $\tau$ | **0.1** | Ambang konflik untuk routing dinamis. Kalibrasi via kuantil $K_{tv}$ validasi (mis. persentil-80); rentang pencarian $[0.05, 0.3]$ |
| Loss | `LAMBDA_FUSED` | $\lambda_f$ | 1.0 | Bobot supervisi opini fusi $L_{\text{fused}}$ |
| Loss | `GAMMA` | $\gamma$ | 1.0 | Bobot Semantic Conflict Loss $L_{\text{con}}$ |
| Evaluasi | `SELECT_METRIC` | — | `"macro_f1"` | Metrik seleksi checkpoint terbaik |
| Evaluasi | `UCE_BINS` | — | 10 | Jumlah bin untuk kalibrasi UCE |

**Prosedur Kalibrasi $\tau$:** (1) Latih model dengan $\tau = 0.1$. (2) Hitung distribusi $K_{tv}$ pada himpunan validasi/test; laporkan kuantil p50/p80/p90/p95/p99. (3) Jika Rute B aktif pada $<1\%$ atau $>50\%$ sampel, geser $\tau$ menuju persentil-80 dan ulangi evaluasi. (4) Laporkan sensitivitas performa terhadap $\tau$ sebagai analisis ablasi tesis.

---

### V. PERSYARATAN TEKNIS IMPLEMENTASI KODE
1.  **Stabilitas Numerik:** Epsilon $10^{-8}$ pada seluruh pembagian; `torch.clamp` untuk input log/gamma/digamma; normalisasi Dempster menggunakan `1 / clamp(1 - K, min=ε)`; $u_{\text{fusion}}$ di-clamp ke $[\epsilon, 1]$; masking token padding dengan $-10^9$ sebelum softmax arah visual→teks.
2.  **Modularitas:** Kelas PyTorch rapi (`nn.Module`): `TextEncoder`, `ImageEncoder`, `BiCoAttention`, `ENNHead`, `ADEFModule`, `EvidentialLoss`. Fungsi utilitas terpisah untuk Subjective Logic (`compute_belief_uncertainty`), konversi opini→Dirichlet (`opinion_to_dirichlet`), dan Semantic Conflict Loss.
3.  **Backbone Beku:** RoBERTa dan DenseNet121 dibekukan (`requires_grad=False` + `torch.no_grad()`) agar adil terhadap 4 baseline dan hemat VRAM.
4.  **Metrik Evaluasi:** Accuracy, Weighted-F1, **Macro-F1**, **F1 per kelas**, classification report lengkap, confusion matrix, **Expected Uncertainty Calibration Error (UCE)**, statistik routing ADEF (% Rute A/B, rata-rata $K_{tv}$, kuantil $K_{tv}$), serta analisis ketidakpastian (mean $u$ prediksi benar vs salah).
5.  **Device-Aware:** `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
6.  **Reprodusibilitas:** Seed tetap (42) untuk `random`, `numpy`, `torch`, dan cuDNN deterministik.

Tuliskan implementasi kode ini dengan lengkap, profesional, terstruktur, dan bersih!

---

### Lampiran A — Catatan Revisi v2: Analisis Akar Masalah & Perbaikan

Hasil eksperimen v1 pada test set: **Accuracy 0.672, Macro-F1 0.432, dan kelas Netral F1 = 0.00 (tidak pernah diprediksi)** — terendah di antara 5 model (baseline lain mencapai Macro-F1 0.55–0.57 dan F1 Netral 0.30–0.34). Rute B ADEF aktif pada **0%** sampel.

| # | Gejala | Akar Masalah (v1) | Perbaikan (v2) |
|---|---|---|---|
| 1 | Kelas Netral tidak pernah diprediksi | Ketimpangan kelas 59/30/10 tanpa kompensasi; loss Sum-of-Squares didominasi kelas mayoritas | Pembobotan kelas $w_y$ pada $L_{\text{err}}$ (§III.1); `USE_CLASS_WEIGHTS` |
| 2 | Checkpoint "terbaik" mengabaikan Netral | Seleksi model memakai Weighted-F1 yang bias ke kelas mayoritas | Seleksi berbasis **Macro-F1** (`SELECT_METRIC`) + laporan F1 per kelas |
| 3 | Jalur Co-Attention lemah | "Co-Attention" v1 beroperasi pada vektor ter-*pooling* $[B, d]$: matriks atensi $A$ berdegenerasi menjadi **skalar** $[B,1,1]$ dengan sigmoid — bukan penyelarasan token–patch | Co-Attention tingkat sekuens sesuai rumus asli: $S = \mathbf{H}_t \mathbf{W} \mathbf{H}_v^T / \sqrt{d} \in \mathbb{R}^{L_t \times N_v}$ dengan softmax dua arah + masking (§II.2) |
| 4 | Rute B tidak pernah aktif (0% sampel) | $\tau = 0.5$ hardcoded, sedangkan rata-rata $K_{tv} \approx 0.014$; data konflik juga terfilter | $\tau$ menjadi hyperparameter terkonfigurasi (`CFG.TAU = 0.1`) + prosedur kalibrasi kuantil (§IV); `FILTER_CONFLICT_PAIRS` sebagai flag ablasi |
| 5 | Keputusan fusi tidak teroptimasi | Loss hanya menyentuh $\alpha_t, \alpha_v, \alpha_c$; $p_{\text{fusion}}$ tidak ada di dalam loss | Supervisi opini fusi $L_{\text{fused}}$ via rekonstruksi $\alpha_{\text{fusion}}$ (§III.2) |
| 6 | Konvergensi lambat (train acc 76% @ epoch 30) | LR $2 \times 10^{-5}$ untuk head acak (LR skala fine-tuning backbone, padahal backbone beku) | `LR = 1 \times 10^{-4}` untuk seluruh parameter terlatih (§IV) |
