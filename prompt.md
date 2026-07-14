# Prompt Rekayasa Kode: Studi Komparatif Integrasi Evidential Deep Learning (EDL) dalam Strategi Fusi Multimodal

Bertindaklah sebagai **Senior AI Engineer & Deep Learning Researcher** yang ahli dalam *Multimodal Sentiment Analysis* (MSA), *Uncertainty Quantification*, dan *PyTorch Development*. Tugas Anda adalah menghasilkan kode PyTorch yang bersih, modular, terdokumentasi dengan baik, dan stabil secara numerik untuk mengevaluasi **empat arsitektur fusi multimodal** yang diintegrasikan dengan **Evidential Deep Learning (EDL)**.

Kode ini akan digunakan untuk eksperimen studi komparatif sub-tesis saya menggunakan dataset MVSA (Teks dan Gambar) secara utuh.

---

## 1. Spesifikasi Teknis Input & Representasi
1. **Input Teks ($X_t$):** Berupa sekuens token teks berdimensi $[B, L_t, d_{BERT}]$ yang diekstrak dari RoBERTa (misal: $d_{BERT} = 768$, panjang sekuens $L_t = 150$).
2. **Input Gambar ($X_v$):** Fitur spasial gambar berdimensi $[B, C_{img}, H, W]$ atau fitur global pooled $[B, d_{CNN}]$ dari DenseNet (misal: $d_{CNN} = 1024$ atau $2048$).
3. **Dimensi Laten Bersama ($d_{proj}$):** Kedua modalitas diproyeksikan ke ruang laten yang sama sebesar $d_{proj} = 512$ menggunakan linear layer + ReLU + Batch Normalization sebelum masuk ke modul fusi.
4. **Target Label ($y$):** 3 kelas sentimen (Positive, Neutral, Negative) dalam format One-Hot Vector $[B, 3]$ atau indeks kelas $[B]$ untuk training.

---

## 2. Arsitektur Model yang Wajib Diimplementasikan

Anda harus membuat kelas-kelas model PyTorch berikut secara modular:

### Model 1: Early Fusion + EDL (`EarlyFusionEDL`)
* **Alur:** Fitur teks dan gambar hasil proyeksi digabungkan secara langsung menggunakan operasi konkatenasi (`torch.cat`) pada dimensi fitur, menghasilkan representasi terpadu $[B, 2 \times d_{proj}]$.
* **ENN Head:** Representasi gabungan diproyeksikan ke dimensi kelas $[B, 3]$ menggunakan fully-connected layer, diikuti oleh fungsi aktivasi non-negatif (`Softplus` atau `ReLU`) untuk menghasilkan nilai bukti (*Evidence*) $e_{early} \in [0, \infty)^3$.
* **Output:** Parameter Dirichlet $\alpha = e_{early} + 1$.

### Model 2: Late Fusion + EDL (`LateFusionEDL`)
* **Alur:** Teks dan gambar diproses secara independen melalui jalurnya masing-masing.
* **ENN Heads:**
  * Jalur Teks menghasilkan bukti $e_t = \text{NonNegative}(W_t h_t + b_t)$ dan parameter Dirichlet $\alpha_t = e_t + 1$.
  * Jalur Gambar menghasilkan bukti $e_v = \text{NonNegative}(W_v h_v + b_v)$ dan parameter Dirichlet $\alpha_v = e_v + 1$.
* **Formulasi Evidensial (Subjective Logic):**
  Untuk masing-masing modalitas $m \in \{t, v\}$, hitung:
  * Kekuatan Dirichlet: $S_m = \sum_{k=1}^3 \alpha_{m, k}$
  * Massa Keyakinan (*Belief Mass*): $b_{m, k} = \frac{e_{m, k}}{S_m}$
  * Massa Ketidakpastian (*Uncertainty Mass*): $u_m = \frac{3}{S_m}$
* **Aturan Kombinasi Dempster (Dempster's Rule of Combination):**
  Gabungkan opini dari Teks ($M_t = \{b_t, u_t\}$) dan Gambar ($M_v = \{b_v, u_v\}$) di tingkat keputusan (*decision-level*):
  * Kalkulasi Faktor Konflik ($C$):
    $$C = \sum_{i \neq j} b_{t, i} \cdot b_{v, j}$$
  * Skala Normalisasi: $1 - C$ (Tambahkan epsilon kecil $\epsilon = 1e-8$ untuk stabilitas jika $C \approx 1$).
  * Massa Fusi ($b_{fusion, k}$) dan Ketidakpastian Fusi ($u_{fusion}$):
    $$b_{fusion, k} = \frac{1}{1-C} \left( b_{t, k} \cdot b_{v, k} + b_{t, k} \cdot u_v + b_{v, k} \cdot u_t \right)$$
    $$u_{fusion} = \frac{1}{1-C} (u_t \cdot u_v)$$
* **Output:** Rekonstruksi parameter Dirichlet gabungan $\alpha_{fusion} = e_{fusion} + 1$, di mana $e_{fusion, k} = \frac{3 \cdot b_{fusion, k}}{u_{fusion}}$.

### Model 3: Cross-Attention Fusion + EDL (`CrossAttentionEDL`)
* **Alur:** Menggunakan mekanisme atensi satu arah (*unidirectional cross-attention*).
* **Mekanisme:** Teks bertindak sebagai *Query* ($Q$), sedangkan gambar bertindak sebagai *Key* ($K$) dan *Value* ($V$).
  * Proyeksi matriks: $Q = h_t W_Q$, $K = h_v W_K$, $V = h_v W_V$.
  * Skor Atensi Spasial:
    $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{proj}}}\right) V$$
* **ENN Head:** Hasil representasi teratensi ini dimasukkan ke dalam ENN Head untuk menghasilkan $e_{cross}$ dan parameter Dirichlet $\alpha_{cross}$.

### Model 4: Co-Attention Fusion + EDL (`CoAttentionEDL`)
* **Alur:** Menggunakan fusi interaktif dua arah (*bidirectional co-attention*).
* **Mekanisme:**
  * **Jalur Teks-ke-Gambar (Text-Guided Visual Attention):** Teks memandu pencarian area gambar yang relevan.
  * **Jalur Gambar-ke-Teks (Visual-Guided Text Attention):** Gambar memandu pencarian kata-kata yang relevan.
  * Gabungkan hasil representasi teratensi dari kedua jalur menggunakan gerbang adaptif atau konkatenasi terproyeksi $[B, d_{proj}]$.
* **ENN Head:** Dimasukkan ke dalam ENN Head untuk menghasilkan $e_{co}$ dan parameter Dirichlet $\alpha_{co}$.

---

## 3. Formulasi Kehilangan (EDL Loss Function)
Untuk setiap model, optimasi parameter dilakukan menggunakan fungsi kerugian probabilistik Dirichlet. Implementasikan kelas `EvidentialLoss` yang mencakup komponen berikut:

1. **Evidential Cross-Entropy Loss ($L_{CE}$):**
   $$L_{CE}(\alpha) = \sum_{k=1}^K y_k \cdot \left( \psi(S) - \psi(\alpha_k) \right)$$
   di mana $\psi(\cdot)$ adalah fungsi *Digamma*.

2. **Kullback-Leibler (KL) Divergence Regularization ($L_{KL}$):**
   Untuk menekan bukti palsu (*misleading evidence*) pada kelas yang salah:
   $$L_{KL}(\alpha) = \log \left( \frac{\Gamma\left(\sum_{k=1}^K \tilde{\alpha}_k\right)}{\Gamma(K) \prod_{k=1}^K \Gamma(\tilde{\alpha}_k)} \right) + \sum_{k=1}^K (\tilde{\alpha}_k - 1) \left( \psi(\tilde{\alpha}_k) - \psi\left(\sum_{j=1}^K \tilde{\alpha}_j\right) \right)$$
   di mana:
   * $\tilde{\alpha} = y + (1 - y) \odot \alpha$ adalah parameter Dirichlet setelah bukti benar dihapus.
   * $\Gamma(\cdot)$ adalah fungsi *Gamma* (`torch.lgamma` wajib digunakan untuk stabilitas numerik).

3. **Total Loss ($L_{total}$):**
   $$L_{total} = L_{CE}(\alpha) + \lambda_t \cdot L_{KL}(\alpha)$$
   di mana $\lambda_t = \min(1.0, \frac{\text{epoch}}{\text{annealing\_epochs}})$ adalah koefisien pemanasan (*annealing coefficient*) untuk mencegah konvergensi dini ke distribusi seragam di awal latihan.

*Catatan untuk Late Fusion:* Karena Late Fusion memiliki cabang Teks murni, Gambar murni, dan Fusi keputusan, hitung loss secara multi-task:
$$L_{total\_late} = L_{total}(\alpha_t) + L_{total}(\alpha_v) + L_{total}(\alpha_{fusion})$$

---

## 4. Metrik Evaluasi Kognitif
Selain akurasi standar dan F1-score, implementasikan fungsi untuk menghitung **Expected Uncertainty Calibration Error (UCE)** untuk membuktikan bahwa model menyadari keraguannya sendiri saat salah klasifikasi:
* Bagi prediksi menjadi $J$ bin berdasarkan tingkat ketidakpastian $u$.
* Hitung selisih absolut antara rata-rata error model dan rata-rata ketidakpastian di setiap bin:
  $$\text{UCE} = \sum_{j=1}^J \frac{|B_j|}{N} \left| \text{err}(B_j) - \text{uncert}(B_j) \right|$$

---

## 5. Persyaratan Kode PyTorch
1. **Stabilitas Numerik:** Gunakan pelindung batas bawah/clamping (`torch.clamp(alpha, min=1e-10)`) sebelum memanggil fungsi log, lgamma, atau digamma untuk mencegah kemunculan `NaN`.
2. **Kemandirian Modul:** Buatlah kode dalam format satu file skrip Python lengkap (`train_comparative.py`) yang berisi definisi arsitektur, perhitungan loss, penghitungan metrik UCE, penyiapan dataset tiruan (*dummy dataset loader*) untuk demonstrasi run, serta loop pelatihan & evaluasi lengkap untuk keempat model.
3. **Format Output:** Berikan penjelasan singkat mengenai struktur kode Anda, diikuti oleh blok kode utuh yang siap dieksekusi tanpa adanya bagian kode yang terpotong.
