
Anda adalah **AI Lead Code Engineer** yang ahli dalam bidang **Multimodal Sentiment Analysis (MSA)**, **Deep Learning**, **Evidential Deep Learning (EDL)**, dan **Teori Probabilitas Dempster-Shafer (DST)**.

Tugas Anda adalah menulis kode implementasi **PyTorch** yang bersih, optimal, terkomentari dengan baik, dan siap dijalankan (*production-ready*) untuk model Tesis S2 saya yang berjudul: **"Adaptive Evidential Fusion (ADEF) with Co-Attention"**.

Arsitektur model ini secara ketat dirancang berdasarkan rancangan resmi proposal tesis saya (Erlangga Dewa Sakti, Telkom University) di bawah bimbingan **Prof. Dr. ADIWIJAYA**. Model ini menggabungkan ekstraksi korelasi silang teks-visual menggunakan **Co-Attention** dengan kerangka kuantifikasi ketidakpastian berbasis **Evidential Deep Learning (EDL)**.

---

### I. SPESIFIKASI INPUT & OUTPUT DATA
1. **Input Teks ($T$):** Tokenized text IDs dengan ukuran `[batch_size, seq_len]` dan Attention Mask `[batch_size, seq_len]`. Panjang sekuens maksimal ($L_t$) adalah **150**.
2. **Input Gambar ($V$):** Tensor gambar RGB dengan ukuran `[batch_size, 3, 224, 224]`.
3. **Output Prediksi ($\hat{y}$):** Label sentimen mayoritas dari **3 kelas** (0: Negatif, 1: Netral, 2: Positif).

---

### II. STRUKTUR ARSITEKTUR MODEL (ADEFCoAttnNet)

Terapkan model PyTorch modular yang terdiri dari komponen-komponen berikut:

#### 1. Unimodal Feature Extraction (Feature Extraction Layer)
*   **Text Encoder ($\mathbf{h}_t$):** Gunakan Pre-trained Language Model `RoBERTa-base` (dari HuggingFace Transformers). Ambil representasi *hidden state* dari token `[CLS]` atau rata-rata pooling. Proyeksikan representasi ini menggunakan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi laten bersama $d = 512$.
    $$\mathbf{h}_t = f_{\text{RoBERTa}}(T)$$
*   **Image Encoder ($\mathbf{h}_v$):** Gunakan model CNN pre-trained `DenseNet121` atau `DenseNet201` (dari Torchvision). Hilangkan lapisan klasifikasi akhir (*classifier head*). Ambil representasi fitur global setelah Global Average Pooling, lalu proyeksikan menggunakan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi laten bersama $d = 512$.
    $$\mathbf{h}_v = f_{\text{DenseNet}}(V)$$

#### 2. Bidirectional Co-Attention Module (Co-Attention Layer)
Modul ini bertugas menangkap interaksi dan penyelarasan semantik halus secara dua arah (*fine-grained bidirectional alignment*) antara kata-kata teks dan area visual gambar sebelum dilakukan estimasi ketidakpastian.
*   **Matriks Bobot Atensi ($A$):** Hitung bobot atensi silang menggunakan perkalian dot-product terproyeksi ($\mathbf{W} \in \mathbb{R}^{d \times d}$):
    $$A = \text{Softmax}\left(\frac{\mathbf{h}_t \mathbf{W} \mathbf{h}_v^T}{\sqrt{d}}\right)$$
*   **Representasi Gabungan ($\mathbf{h}_c$):** Lakukan perkalian matriks untuk menyelaraskan fitur, lalu gabungkan (*concatenate*) representasi teratensi dari kedua modalitas:
    $$\mathbf{h}_c = \text{Concat}(\mathbf{h}_t \cdot A, \mathbf{h}_v \cdot A^T)$$
    $\mathbf{h}_c$ bertindak sebagai jalur ketiga yang merepresentasikan hubungan interaksi teks-gambar.

#### 3. Evidential Neural Network (ENN) Heads
Alih-alih menggunakan aktivasi Softmax deterministik di tahap akhir yang sering menyebabkan masalah *overconfidence* pada prediksi yang salah, model ini melewatkan ketiga fitur ($\mathbf{h}_t$, $\mathbf{h}_v$, dan $\mathbf{h}_c$) secara paralel ke tiga ENN Head independen (Text ENN, Image ENN, Co-Attention ENN).
*   **Ekstraksi Bukti (*Evidence Extraction*):** Gunakan lapisan fully-connected dengan fungsi aktivasi non-negatif seperti **ReLU** (atau Softplus) untuk menjamin nilai bukti $e \ge 0$.
    Untuk setiap jalur $k \in \{t, v, c\}$ dan kelas $i \in \{1, 2, 3\}$:
    $$e_{k,i} = \text{ReLU}(\mathbf{W}_k \mathbf{h}_k + \mathbf{b}_k)$$
*   **Parameter Distribusi Dirichlet ($\alpha$):**
    $$\alpha_{k,i} = e_{k,i} + 1$$
*   **Kekuatan Bukti Total ($S_k$):**
    $$S_k = \sum_{i=1}^M \alpha_{k,i}$$
    di mana $M = 3$ (jumlah kelas sentimen).

#### 4. Kuantifikasi Belief Mass & Uncertainty Mass
Berdasarkan *Subjective Logic* (SL), hitung massa keyakinan (*Belief*) dan massa ketidakpastian global (*Uncertainty*) secara independen untuk masing-masing dari ketiga jalur:
*   **Belief Mass ($b_{k,i}$):** Representasi seberapa kuat bukti yang mendukung kelas $i$.
    $$b_{k,i} = \frac{e_{k,i}}{S_k}$$
*   **Uncertainty Mass ($u_k$):** Mengukur keraguan model akibat kurangnya bukti atau adanya noise/distorsi data.
    $$u_k = \frac{M}{S_k}$$
*   Sesuai dengan aksioma probabilitas Subjective Logic, pastikan:
    $$\sum_{i=1}^M b_{k,i} + u_k = 1$$

#### 5. Modul Adaptive Evidential Fusion (ADEF)
Modul ini bertugas melakukan fusi adaptif secara dinamis dengan mengevaluasi tingkat perselisihan opini antara jalur teks murni dan gambar murni.
*   **Kalkulasi Massa Konflik ($K_{tv}$):** Hitung tingkat kontradiksi opini antara modalitas Teks ($t$) dan Gambar ($v$). Gunakan formula matematis berikut (Double Sigma untuk mencegah ambiguitas indeks):
    $$K_{tv} = \sum_{i=1}^M \sum_{\substack{j=1 \\ j \neq i}}^M b_{t,i} \cdot b_{v,j}$$
*   **Mekanisme Peralihan Rute Dinamis (*Dynamic Routing Decision*):** Bandingkan $K_{tv}$ dengan ambang batas toleransi konflik ($\tau$, misalnya $\tau = 0.5$):

    *   **Rute A: Fusi Normal (Dempster-Shafer Standard) — Jika $K_{tv} \le \tau$:**
        Kondisi ini menunjukkan teks dan gambar selaras. Gunakan aturan kombinasi Dempster secara asosiatif dalam dua tahap:
        *   *Tahap 1 (Fusi Teks-Visual):* Gabungkan keyakinan dari Teks ($t$) dan Gambar ($v$).
            $$b_{tv,i} = \frac{1}{1 - K_{tv}} (b_{t,i} \cdot b_{v,i} + b_{t,i} \cdot u_v + b_{v,i} \cdot u_t)$$
            $$u_{tv} = \frac{1}{1 - K_{tv}} (u_t \cdot u_v)$$
        *   *Tahap 2 (Konsensus Akhir / Sinergi dengan Co-Attention):* Gabungkan hasil tahap 1 dengan jalur Co-Attention ($c$) sebagai penguat keyakinan (*confidence booster*). Hitung massa konflik baru $K_{tvc}$:
            $$K_{tvc} = \sum_{i=1}^M \sum_{\substack{j=1 \\ j \neq i}}^M b_{tv,i} \cdot b_{c,j}$$
            $$b_{\text{fusion},i} = \frac{1}{1 - K_{tvc}} (b_{tv,i} \cdot b_{c,i} + b_{tv,i} \cdot u_c + b_{c,i} \cdot u_{tv})$$
            $$u_{\text{fusion}} = \frac{1}{1 - K_{tvc}} (u_{tv} \cdot u_c)$$

    *   **Rute B: Fusi Resolusi Konflik (Conflict-Aware Fusion) — Jika $K_{tv} > \tau$:**
        Kondisi ini menandakan kontradiksi ekstrem (misal: sarkasme). Aturan ortogonal Dempster standar akan gagal (*Zadeh's Paradox*). Bypass fusi normal, gunakan $K_{tv}$ sebagai tuas pengontrol dinamis untuk meredam opini unimodal yang bertentangan, dan alihkan dominasi bobot sepenuhnya kepada representasi relasional tingkat tinggi dari modul **Co-Attention**:
        $$b_{\text{fusion},i} = (1 - K_{tv}) \cdot \left( \frac{b_{t,i} + b_{v,i}}{2} \right) + K_{tv} \cdot b_{c,i}$$
        Ekstrak sisa ruang ketidakpastian yang tersisa:
        $$u_{\text{fusion}} = 1 - \sum_{i=1}^M b_{\text{fusion},i}$$

#### 6. Final Decision Making
Sistem tidak membuang nilai ketidakpastian final ($u_{\text{fusion}}$). Distribusikan keraguan tersebut secara adil ke seluruh kelas sebagai prior seragam:
*   **Final Expectation Probability ($p_i$):**
    $$p_i = b_{\text{fusion},i} + \frac{u_{\text{fusion}}}{M}$$
*   **Output Prediksi Sentimen Final ($\hat{y}$):**
    $$\hat{y} = \text{Argmax}(p_1, p_2, p_3)$$

---

### III. BLENDED LOSS FUNCTION (TRAINING OBJECTIVE)

Model harus dilatih menggunakan pendekatan *Multi-Task Learning* dengan meminimalkan gabungan kerugian klasifikasi evidensial dan penalti konflik:

#### 1. Multi-Task Evidential Loss ($L_{\text{sup}}$)
Hitung loss klasifikasi EDL pada ketiga jalur secara paralel untuk memastikan ketiga representasi dilatih dengan baik:
$$L_{\text{sup}} = L(\alpha_t) + L(\alpha_v) + L(\alpha_c)$$
Di mana untuk setiap jalur $k$, fungsi loss $L(\alpha_k)$ dirumuskan sebagai penggabungan Bayes Risk dengan Sum of Squares Loss dan regularisasi KL-Divergence:
$$L(\alpha_k) = L_{\text{err}}(\alpha_k) + \lambda_t L_{\text{KL}}(\alpha_k)$$
*   **Error Term ($L_{\text{err}}$):**
    $$L_{\text{err}}(\alpha_{k}) = \sum_{j=1}^M (y_j - \hat{p}_{k,j})^2 + \frac{\hat{p}_{k,j}(1 - \hat{p}_{k,j})}{S_k + 1}$$
    di mana $y$ adalah target label berupa *one-hot vector*, dan $\hat{p}_{k,j} = \alpha_{k,j} / S_k$.
*   **Regularisasi KL-Divergence ($L_{\text{KL}}$):** Penalti divergensi untuk menekan bukti pada kelas yang salah agar tidak menghasilkan prediksi *overconfident*:
    $$L_{\text{KL}}(\alpha_k) = \log \left( \frac{\Gamma(\sum_{j=1}^M \tilde{\alpha}_{k,j})}{\Gamma(M) \prod_{j=1}^M \Gamma(\tilde{\alpha}_{k,j})} \right) + \sum_{j=1}^M (\tilde{\alpha}_{k,j} - 1) \left[ \psi(\tilde{\alpha}_{k,j}) - \psi\left(\sum_{l=1}^M \tilde{\alpha}_{l}\right) \right]$$
    di mana $\tilde{\alpha}_{k} = y + (1 - y) \odot \alpha_k$, $\Gamma(\cdot)$ adalah fungsi Gamma, dan $\psi(\cdot)$ adalah fungsi Digamma.
*   **Annealing Coefficient ($\lambda_t$):** Nilai peningkatan porsi regularisasi secara bertahap berdasarkan epoch pelatihan saat ini ($t$) untuk menghindari konvergensi dini:
    $$\lambda_t = \min(1.0, \frac{t}{10})$$

#### 2. Semantic Conflict Loss ($L_{\text{con}}$)
Untuk memandu enkoder agar dapat mendeteksi inkongruensi emosi (sarkasme) dan memetakan konflik semantik langsung menjadi ketidakpastian multimodal:
$$L_{\text{con}} = d_{\text{PD}} \cdot d_{\text{CC}} = \frac{1}{2} (1 - u_t) (1 - u_v) \sum_{i=1}^M |p_{t,i} - p_{v,i}|$$
*   $p_{t,i}$ dan $p_{v,i}$ adalah probabilitas ekspektasi unimodal teks dan gambar.
*   $u_t$ dan $u_v$ adalah ketidakpastian unimodal teks dan gambar.

#### 3. Total Loss Keseluruhan Jaringan
$$L_{\text{overall}} = L_{\text{sup}} + \gamma L_{\text{con}}$$
di mana $\gamma$ adalah hyperparameter balancing (default: 1.0).

---

### IV. PERSYARATAN TEKNIS IMPLEMENTASI KODE
1.  **Stabilitas Numerik:** Terapkan pencegahan pembagian dengan nol (penambahan epsilon $1e-8$) dan gunakan `torch.clamp` untuk menjaga nilai input log/gamma agar terhindar dari *NaN* atau *gradient explosion/vanishing*.
2.  **Modularitas:** Tulis kode dalam kelas PyTorch yang rapi (`torch.nn.Module`). Pisahkan fungsi perhitungan kombinasi Dempster-Shafer, perutean ADEF, Evidential Loss, dan Conflict Loss ke dalam fungsi utilitas atau modul khusus agar mudah di-abstraksi.
3.  **Metrik Evaluasi:** Tuliskan juga fungsi utilitas untuk menghitung metrik performa standar: **Accuracy**, **Macro F1-Score**, dan metrik kalibrasi ketidakpastian **Expected Uncertainty Calibration Error (UCE)** untuk membuktikan keandalan model Anda dalam mengukur keraguan dirinya sendiri secara kuantitatif.
4.  **Device-Aware:** Pastikan kode mendukung pelatihan paralel menggunakan GPU (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`).

Tuliskan implementasi kode ini dengan lengkap, profesional, terstruktur, dan bersih!
