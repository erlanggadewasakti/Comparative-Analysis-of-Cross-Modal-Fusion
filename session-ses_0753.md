# Fix ADEF neutral mispredictions & tau config

**Session ID:** ses_0753e645bffeeY7ZKEpHWEysJS
**Created:** 7/22/2026, 11:56:11 PM
**Updated:** 7/23/2026, 12:24:36 AM

---

## User

& 'd:\Coding\Project\All Thesis\Comparative Analysis of Cross-Modal Fusion\adef_co_attention.ipynb' fix my adef arcithecture, why it can miss predicted in neutral. and also make the tau as a config, do for all hyperparameter. my plan is from & 'd:\Coding\Project\All Thesis\Comparative Analysis of Cross-Modal Fusion\rancangan_tesis_adef_co_attention.md' but i don't know this is optimal or not. you can free to modify and give the best proposed model. just modify the old plan  

---

## Assistant (Plan · Kimi K3 (2x usage) · 16.3s)

_Thinking:_

The user wants me to:
1. Analyze their ADEF co-attention architecture in the notebook `adef_co_attention.ipynb` to figure out why it mispredicts on "neutral" class
2. Make tau (temperature) a config parameter, and do the same for all hyperparameters
3. Their plan is in `rancangan_tesis_adef_co_attention.md` - they want me to review it, determine if it's optimal, and propose the best model by modifying the old plan

I'm in plan mode - read-only. I need to read both files first, analyze, and construct a plan. Let me read the notebook and the markdown plan.

Let me start by reading both files. The notebook could be large, so let me read it carefully.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\rancangan_tesis_adef_co_attention.md"
}
```

**Output:**
```
<path>D:\Coding\Project\All Thesis\Comparative Analysis of Cross-Modal Fusion\rancangan_tesis_adef_co_attention.md</path>
<type>file</type>
<content>
1: 
2: Anda adalah **AI Lead Code Engineer** yang ahli dalam bidang **Multimodal Sentiment Analysis (MSA)**, **Deep Learning**, **Evidential Deep Learning (EDL)**, dan **Teori Probabilitas Dempster-Shafer (DST)**.
3: 
4: Tugas Anda adalah menulis kode implementasi **PyTorch** yang bersih, optimal, terkomentari dengan baik, dan siap dijalankan (*production-ready*) untuk model Tesis S2 saya yang berjudul: **"Adaptive Evidential Fusion (ADEF) with Co-Attention"**.
5: 
6: Arsitektur model ini secara ketat dirancang berdasarkan rancangan resmi proposal tesis saya (Erlangga Dewa Sakti, Telkom University) di bawah bimbingan **Prof. Dr. ADIWIJAYA**. Model ini menggabungkan ekstraksi korelasi silang teks-visual menggunakan **Co-Attention** dengan kerangka kuantifikasi ketidakpastian berbasis **Evidential Deep Learning (EDL)**.
7: 
8: ---
9: 
10: ### I. SPESIFIKASI INPUT & OUTPUT DATA
11: 1. **Input Teks ($T$):** Tokenized text IDs dengan ukuran `[batch_size, seq_len]` dan Attention Mask `[batch_size, seq_len]`. Panjang sekuens maksimal ($L_t$) adalah **150**.
12: 2. **Input Gambar ($V$):** Tensor gambar RGB dengan ukuran `[batch_size, 3, 224, 224]`.
13: 3. **Output Prediksi ($\hat{y}$):** Label sentimen mayoritas dari **3 kelas** (0: Negatif, 1: Netral, 2: Positif).
14: 
15: ---
16: 
17: ### II. STRUKTUR ARSITEKTUR MODEL (ADEFCoAttnNet)
18: 
19: Terapkan model PyTorch modular yang terdiri dari komponen-komponen berikut:
20: 
21: #### 1. Unimodal Feature Extraction (Feature Extraction Layer)
22: *   **Text Encoder ($\mathbf{h}_t$):** Gunakan Pre-trained Language Model `RoBERTa-base` (dari HuggingFace Transformers). Ambil representasi *hidden state* dari token `[CLS]` atau rata-rata pooling. Proyeksikan representasi ini menggunakan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi laten bersama $d = 512$.
23:     $$\mathbf{h}_t = f_{\text{RoBERTa}}(T)$$
24: *   **Image Encoder ($\mathbf{h}_v$):** Gunakan model CNN pre-trained `DenseNet121` atau `DenseNet201` (dari Torchvision). Hilangkan lapisan klasifikasi akhir (*classifier head*). Ambil representasi fitur global setelah Global Average Pooling, lalu proyeksikan menggunakan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi laten bersama $d = 512$.
25:     $$\mathbf{h}_v = f_{\text{DenseNet}}(V)$$
26: 
27: #### 2. Bidirectional Co-Attention Module (Co-Attention Layer)
28: Modul ini bertugas menangkap interaksi dan penyelarasan semantik halus secara dua arah (*fine-grained bidirectional alignment*) antara kata-kata teks dan area visual gambar sebelum dilakukan estimasi ketidakpastian.
29: *   **Matriks Bobot Atensi ($A$):** Hitung bobot atensi silang menggunakan perkalian dot-product terproyeksi ($\mathbf{W} \in \mathbb{R}^{d \times d}$):
30:     $$A = \text{Softmax}\left(\frac{\mathbf{h}_t \mathbf{W} \mathbf{h}_v^T}{\sqrt{d}}\right)$$
31: *   **Representasi Gabungan ($\mathbf{h}_c$):** Lakukan perkalian matriks untuk menyelaraskan fitur, lalu gabungkan (*concatenate*) representasi teratensi dari kedua modalitas:
32:     $$\mathbf{h}_c = \text{Concat}(\mathbf{h}_t \cdot A, \mathbf{h}_v \cdot A^T)$$
33:     $\mathbf{h}_c$ bertindak sebagai jalur ketiga yang merepresentasikan hubungan interaksi teks-gambar.
34: 
35: #### 3. Evidential Neural Network (ENN) Heads
36: Alih-alih menggunakan aktivasi Softmax deterministik di tahap akhir yang sering menyebabkan masalah *overconfidence* pada prediksi yang salah, model ini melewatkan ketiga fitur ($\mathbf{h}_t$, $\mathbf{h}_v$, dan $\mathbf{h}_c$) secara paralel ke tiga ENN Head independen (Text ENN, Image ENN, Co-Attention ENN).
37: *   **Ekstraksi Bukti (*Evidence Extraction*):** Gunakan lapisan fully-connected dengan fungsi aktivasi non-negatif seperti **ReLU** (atau Softplus) untuk menjamin nilai bukti $e \ge 0$.
38:     Untuk setiap jalur $k \in \{t, v, c\}$ dan kelas $i \in \{1, 2, 3\}$:
39:     $$e_{k,i} = \text{ReLU}(\mathbf{W}_k \mathbf{h}_k + \mathbf{b}_k)$$
40: *   **Parameter Distribusi Dirichlet ($\alpha$):**
41:     $$\alpha_{k,i} = e_{k,i} + 1$$
42: *   **Kekuatan Bukti Total ($S_k$):**
43:     $$S_k = \sum_{i=1}^M \alpha_{k,i}$$
44:     di mana $M = 3$ (jumlah kelas sentimen).
45: 
46: #### 4. Kuantifikasi Belief Mass & Uncertainty Mass
47: Berdasarkan *Subjective Logic* (SL), hitung massa keyakinan (*Belief*) dan massa ketidakpastian global (*Uncertainty*) secara independen untuk masing-masing dari ketiga jalur:
48: *   **Belief Mass ($b_{k,i}$):** Representasi seberapa kuat bukti yang mendukung kelas $i$.
49:     $$b_{k,i} = \frac{e_{k,i}}{S_k}$$
50: *   **Uncertainty Mass ($u_k$):** Mengukur keraguan model akibat kurangnya bukti atau adanya noise/distorsi data.
51:     $$u_k = \frac{M}{S_k}$$
52: *   Sesuai dengan aksioma probabilitas Subjective Logic, pastikan:
53:     $$\sum_{i=1}^M b_{k,i} + u_k = 1$$
54: 
55: #### 5. Modul Adaptive Evidential Fusion (ADEF)
56: Modul ini bertugas melakukan fusi adaptif secara dinamis dengan mengevaluasi tingkat perselisihan opini antara jalur teks murni dan gambar murni.
57: *   **Kalkulasi Massa Konflik ($K_{tv}$):** Hitung tingkat kontradiksi opini antara modalitas Teks ($t$) dan Gambar ($v$). Gunakan formula matematis berikut (Double Sigma untuk mencegah ambiguitas indeks):
58:     $$K_{tv} = \sum_{i=1}^M \sum_{\substack{j=1 \\ j \neq i}}^M b_{t,i} \cdot b_{v,j}$$
59: *   **Mekanisme Peralihan Rute Dinamis (*Dynamic Routing Decision*):** Bandingkan $K_{tv}$ dengan ambang batas toleransi konflik ($\tau$, misalnya $\tau = 0.5$):
60: 
61:     *   **Rute A: Fusi Normal (Dempster-Shafer Standard) — Jika $K_{tv} \le \tau$:**
62:         Kondisi ini menunjukkan teks dan gambar selaras. Gunakan aturan kombinasi Dempster secara asosiatif dalam dua tahap:
63:         *   *Tahap 1 (Fusi Teks-Visual):* Gabungkan keyakinan dari Teks ($t$) dan Gambar ($v$).
64:             $$b_{tv,i} = \frac{1}{1 - K_{tv}} (b_{t,i} \cdot b_{v,i} + b_{t,i} \cdot u_v + b_{v,i} \cdot u_t)$$
65:             $$u_{tv} = \frac{1}{1 - K_{tv}} (u_t \cdot u_v)$$
66:         *   *Tahap 2 (Konsensus Akhir / Sinergi dengan Co-Attention):* Gabungkan hasil tahap 1 dengan jalur Co-Attention ($c$) sebagai penguat keyakinan (*confidence booster*). Hitung massa konflik baru $K_{tvc}$:
67:             $$K_{tvc} = \sum_{i=1}^M \sum_{\substack{j=1 \\ j \neq i}}^M b_{tv,i} \cdot b_{c,j}$$
68:             $$b_{\text{fusion},i} = \frac{1}{1 - K_{tvc}} (b_{tv,i} \cdot b_{c,i} + b_{tv,i} \cdot u_c + b_{c,i} \cdot u_{tv})$$
69:             $$u_{\text{fusion}} = \frac{1}{1 - K_{tvc}} (u_{tv} \cdot u_c)$$
70: 
71:     *   **Rute B: Fusi Resolusi Konflik (Conflict-Aware Fusion) — Jika $K_{tv} > \tau$:**
72:         Kondisi ini menandakan kontradiksi ekstrem (misal: sarkasme). Aturan ortogonal Dempster standar akan gagal (*Zadeh's Paradox*). Bypass fusi normal, gunakan $K_{tv}$ sebagai tuas pengontrol dinamis untuk meredam opini unimodal yang bertentangan, dan alihkan dominasi bobot sepenuhnya kepada representasi relasional tingkat tinggi dari modul **Co-Attention**:
73:         $$b_{\text{fusion},i} = (1 - K_{tv}) \cdot \left( \frac{b_{t,i} + b_{v,i}}{2} \right) + K_{tv} \cdot b_{c,i}$$
74:         Ekstrak sisa ruang ketidakpastian yang tersisa:
75:         $$u_{\text{fusion}} = 1 - \sum_{i=1}^M b_{\text{fusion},i}$$
76: 
77: #### 6. Final Decision Making
78: Sistem tidak membuang nilai ketidakpastian final ($u_{\text{fusion}}$). Distribusikan keraguan tersebut secara adil ke seluruh kelas sebagai prior seragam:
79: *   **Final Expectation Probability ($p_i$):**
80:     $$p_i = b_{\text{fusion},i} + \frac{u_{\text{fusion}}}{M}$$
81: *   **Output Prediksi Sentimen Final ($\hat{y}$):**
82:     $$\hat{y} = \text{Argmax}(p_1, p_2, p_3)$$
83: 
84: ---
85: 
86: ### III. BLENDED LOSS FUNCTION (TRAINING OBJECTIVE)
87: 
88: Model harus dilatih menggunakan pendekatan *Multi-Task Learning* dengan meminimalkan gabungan kerugian klasifikasi evidensial dan penalti konflik:
89: 
90: #### 1. Multi-Task Evidential Loss ($L_{\text{sup}}$)
91: Hitung loss klasifikasi EDL pada ketiga jalur secara paralel untuk memastikan ketiga representasi dilatih dengan baik:
92: $$L_{\text{sup}} = L(\alpha_t) + L(\alpha_v) + L(\alpha_c)$$
93: Di mana untuk setiap jalur $k$, fungsi loss $L(\alpha_k)$ dirumuskan sebagai penggabungan Bayes Risk dengan Sum of Squares Loss dan regularisasi KL-Divergence:
94: $$L(\alpha_k) = L_{\text{err}}(\alpha_k) + \lambda_t L_{\text{KL}}(\alpha_k)$$
95: *   **Error Term ($L_{\text{err}}$):**
96:     $$L_{\text{err}}(\alpha_{k}) = \sum_{j=1}^M (y_j - \hat{p}_{k,j})^2 + \frac{\hat{p}_{k,j}(1 - \hat{p}_{k,j})}{S_k + 1}$$
97:     di mana $y$ adalah target label berupa *one-hot vector*, dan $\hat{p}_{k,j} = \alpha_{k,j} / S_k$.
98: *   **Regularisasi KL-Divergence ($L_{\text{KL}}$):** Penalti divergensi untuk menekan bukti pada kelas yang salah agar tidak menghasilkan prediksi *overconfident*:
99:     $$L_{\text{KL}}(\alpha_k) = \log \left( \frac{\Gamma(\sum_{j=1}^M \tilde{\alpha}_{k,j})}{\Gamma(M) \prod_{j=1}^M \Gamma(\tilde{\alpha}_{k,j})} \right) + \sum_{j=1}^M (\tilde{\alpha}_{k,j} - 1) \left[ \psi(\tilde{\alpha}_{k,j}) - \psi\left(\sum_{l=1}^M \tilde{\alpha}_{l}\right) \right]$$
100:     di mana $\tilde{\alpha}_{k} = y + (1 - y) \odot \alpha_k$, $\Gamma(\cdot)$ adalah fungsi Gamma, dan $\psi(\cdot)$ adalah fungsi Digamma.
101: *   **Annealing Coefficient ($\lambda_t$):** Nilai peningkatan porsi regularisasi secara bertahap berdasarkan epoch pelatihan saat ini ($t$) untuk menghindari konvergensi dini:
102:     $$\lambda_t = \min(1.0, \frac{t}{10})$$
103: 
104: #### 2. Semantic Conflict Loss ($L_{\text{con}}$)
105: Untuk memandu enkoder agar dapat mendeteksi inkongruensi emosi (sarkasme) dan memetakan konflik semantik langsung menjadi ketidakpastian multimodal:
106: $$L_{\text{con}} = d_{\text{PD}} \cdot d_{\text{CC}} = \frac{1}{2} (1 - u_t) (1 - u_v) \sum_{i=1}^M |p_{t,i} - p_{v,i}|$$
107: *   $p_{t,i}$ dan $p_{v,i}$ adalah probabilitas ekspektasi unimodal teks dan gambar.
108: *   $u_t$ dan $u_v$ adalah ketidakpastian unimodal teks dan gambar.
109: 
110: #### 3. Total Loss Keseluruhan Jaringan
111: $$L_{\text{overall}} = L_{\text{sup}} + \gamma L_{\text{con}}$$
112: di mana $\gamma$ adalah hyperparameter balancing (default: 1.0).
113: 
114: ---
115: 
116: ### IV. PERSYARATAN TEKNIS IMPLEMENTASI KODE
117: 1.  **Stabilitas Numerik:** Terapkan pencegahan pembagian dengan nol (penambahan epsilon $1e-8$) dan gunakan `torch.clamp` untuk menjaga nilai input log/gamma agar terhindar dari *NaN* atau *gradient explosion/vanishing*.
118: 2.  **Modularitas:** Tulis kode dalam kelas PyTorch yang rapi (`torch.nn.Module`). Pisahkan fungsi perhitungan kombinasi Dempster-Shafer, perutean ADEF, Evidential Loss, dan Conflict Loss ke dalam fungsi utilitas atau modul khusus agar mudah di-abstraksi.
119: 3.  **Metrik Evaluasi:** Tuliskan juga fungsi utilitas untuk menghitung metrik performa standar: **Accuracy**, **Macro F1-Score**, dan metrik kalibrasi ketidakpastian **Expected Uncertainty Calibration Error (UCE)** untuk membuktikan keandalan model Anda dalam mengukur keraguan dirinya sendiri secara kuantitatif.
120: 4.  **Device-Aware:** Pastikan kode mendukung pelatihan paralel menggunakan GPU (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`).
121: 
122: Tuliskan implementasi kode ini dengan lengkap, profesional, terstruktur, dan bersih!

(End of file - total 122 lines)
</content>
```

**Tool: bash**

**Input:**
```json
{
  "command": "python -c \"import json; nb=json.load(open(r'D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb', encoding='utf-8')); print('cells:', len(nb['cells'])); [print(i, c['cell_type'], len(''.join(c['source']))) for i,c in enumerate(nb['cells'])]\""
}
```

**Output:**
```
[31;1mpython: [31;1mThe term 'python' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

```

---

## Assistant (Plan · Kimi K3 (2x usage) · 9.2s)

_Thinking:_

Python is not available directly. Let me check the notebook file size and read it directly with the Read tool. The notebook is JSON so I can read it but it might be big. Let me check file size first, then read it.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-Item \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | Select-Object Length; Get-ChildItem \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\" | Select-Object Name, Length"
}
```

**Output:**
```

[32;1mLength[0m
[32;1m------[0m
343286
     4
222638
343286
255936
256754
264013
262413
  7764
 10847


```

---

## Assistant (Plan · Kimi K3 (2x usage) · 10.2s)

_Thinking:_

The output is truncated oddly. Let me list the files properly.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\" | ForEach-Object { \"$($_.Name) - $($_.Length)\" }"
}
```

**Output:**
```
.gitignore - 4
adef_co_attention_results.png - 222638
adef_co_attention.ipynb - 343286
co_attention.ipynb - 255936
cross_attention.ipynb - 256754
early_fusion.ipynb - 264013
late_fusion.ipynb - 262413
prompt.md - 7764
rancangan_tesis_adef_co_attention.md - 10847

```

---

## Assistant (Plan · Kimi K3 (2x usage) · 9.6s)

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | ConvertFrom-Json; \"Total cells: $($nb.cells.Count)\"; $i=0; foreach ($c in $nb.cells) { $src = ($c.source -join ''); $len = $src.Length; $out = ($c.outputs | Measure-Object).Count; \"$i`t$($c.cell_type)`tlen=$len`toutputs=$out\"; $i++ }"
}
```

**Output:**
```
Total cells: 15
0	markdown	len=59	outputs=0
1	code	len=1254	outputs=2
2	code	len=1078	outputs=1
3	code	len=2331	outputs=2
4	code	len=3428	outputs=2
5	code	len=2131	outputs=1
6	code	len=2001	outputs=1
7	code	len=1314	outputs=1
8	code	len=3395	outputs=1
9	code	len=3104	outputs=2
10	code	len=3246	outputs=1
11	code	len=1317	outputs=1
12	code	len=5198	outputs=61
13	code	len=3106	outputs=4
14	code	len=4098	outputs=2

```

---

## Assistant (Plan · Kimi K3 (2x usage) · 9.3s)

_Thinking:_

Let me extract the source of each code cell to a readable form. I'll dump the sources of all cells to inspect. Since the notebook JSON is 343KB, most of that is outputs. Let me extract just the sources.

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | ConvertFrom-Json; $i=0; foreach ($c in $nb.cells) { if ($c.cell_type -eq 'code') { \"===== CELL $i =====\"; ($c.source -join '') }; $i++ }"
}
```

**Output:**
```
===== CELL 1 =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models
from transformers import RobertaTokenizer, RobertaModel

from PIL import Image
import pandas as pd
import numpy as np
import os
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("\u2705 Imports loaded & seed set.")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

===== CELL 2 =====
# ============================================================
# CONFIGURATION
# ============================================================

class CFG:

    # =========================
    # PATH
    # =========================
    ROOT_DIR = r"D:/MVSA_SINGLE"
    DATA_DIR = r"D:/MVSA_SINGLE/data"
    LABEL_PATH = r"D:/MVSA_SINGLE/labelResultAllFinal.txt"

    # =========================
    # DEVICE
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # HYPERPARAMETERS
    # =========================
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 2e-5
    WEIGHT_DECAY = 1e-4
    MAX_LEN = 150
    D_BERT = 768
    D_CNN = 1024
    D_PROJ = 512
    NUM_CLASSES = 3
    ANNEALING_EPOCHS = 10  # KL reaches full strength at epoch 10 (1/3 of training)
    SEED = 42
    DROPOUT = 0.3

    # =========================
    # PRETRAINED MODELS
    # =========================
    TEXT_MODEL = "roberta-base"
    IMAGE_MODEL = "densenet121"

print(f"\u2705 Configuration loaded. Device: {CFG.DEVICE}")

===== CELL 3 =====
import pandas as pd
import os
# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CFG.LABEL_PATH, header=0, sep=",")
df.columns = ["id", "text_label", "image_label", "final_label"]

def is_valid(row):

    if row["text_label"] == "positive" and row["image_label"] == "negative":
        return False

    if row["text_label"] == "negative" and row["image_label"] == "positive":
        return False

    return True

df = df[df.apply(is_valid, axis=1)]
df = df.reset_index(drop=True)

print(f"Dataset size after filtering: {len(df)}")

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df["label"] = df["final_label"].map(label_map)

# Load text file dengan better error handling
def load_text(sample_id):
    path = os.path.join(CFG.DATA_DIR, f"{sample_id}.txt")

    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                text = f.read().strip()
                if text:  # Jika text berhasil dibaca dan tidak kosong
                    return text
        except FileNotFoundError:
            continue
        except Exception as e:
            continue

    # Jika semua encoding gagal atau file tidak ada
    return ""

# Track failed samples for debugging
failed_samples = []

df["text"] = df["id"].apply(load_text)

# Hitung empty text
empty_text_count = (df["text"] == "").sum()
print(f"\n{'='*60}")
print(f"PREPROCESSING STATISTICS:")
print(f"{'='*60}")
print(f"Total samples: {len(df)}")
print(f"Samples with EMPTY text: {empty_text_count}")
print(f"Samples with VALID text: {len(df) - empty_text_count}")
print(f"Percentage of empty text: {(empty_text_count/len(df)*100):.2f}%")
print(f"{'='*60}\n")

if empty_text_count > 0:
    print("IDs with empty text:")
    empty_ids = df[df["text"] == ""]["id"].tolist()
    for idx in empty_ids[:10]:  # Show first 10
        print(f"  - {idx}")
    if len(empty_ids) > 10:
        print(f"  ... and {len(empty_ids) - 10} more")
    print()

# image path
df["image_path"] = df["id"].apply(
    lambda x: os.path.join(CFG.DATA_DIR, f"{x}.jpg")
)

df.head()

===== CELL 4 =====
# ============================================================
# PYTORCH DATASET & DATALOADERS
# ============================================================

# MVSADataset: PyTorch Dataset for MVSA multimodal sentiment analysis.
class MVSADataset(Dataset):

    def __init__(self, dataframe, tokenizer, transform, max_len=150):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        image_path = row["image_path"]
        label = int(row["label"])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Load and transform image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# INITIALIZE TOKENIZER & TRANSFORMS
# ============================================================
tokenizer = RobertaTokenizer.from_pretrained(CFG.TEXT_MODEL)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# STRATIFIED SPLIT: 70% train, 15% val, 15% test
# ============================================================
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=CFG.SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=CFG.SEED
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"\nTrain label distribution:")
print(train_df['label'].value_counts().sort_index())
print(f"\nVal label distribution:")
print(val_df['label'].value_counts().sort_index())
print(f"\nTest label distribution:")
print(test_df['label'].value_counts().sort_index())

# Create datasets
train_dataset = MVSADataset(train_df, tokenizer, image_transform, CFG.MAX_LEN)
val_dataset = MVSADataset(val_df, tokenizer, image_transform, CFG.MAX_LEN)
test_dataset = MVSADataset(test_df, tokenizer, image_transform, CFG.MAX_LEN)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\n\u2705 DataLoaders created.")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

===== CELL 5 =====
# ============================================================
# FEATURE EXTRACTORS (Pooled for ADEF)
# ============================================================

# RoBERTa text encoder -> CLS pooling -> projection to d_proj
# Input:  input_ids [B, L_t], attention_mask [B, L_t]
# Output: projected [B, d_proj]
class TextEncoder(nn.Module):

    def __init__(self, d_bert=768, d_proj=512):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(CFG.TEXT_MODEL)
        # Freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(d_bert, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        projected = self.projection(cls_output)  # [B, d_proj]
        return projected


# DenseNet-121 image encoder -> GAP -> projection to d_proj
# Input:  image [B, 3, 224, 224]
# Output: projected [B, d_proj]
class ImageEncoder(nn.Module):

    def __init__(self, d_cnn=1024, d_proj=512):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = densenet.features
        # Freeze DenseNet parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(d_cnn, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)  # [B, 1024, H, W]
        pooled = self.pool(features).squeeze(-1).squeeze(-1)  # [B, 1024]
        projected = self.projection(pooled)  # [B, d_proj]
        return projected


print("\u2705 TextEncoder & ImageEncoder defined.")

===== CELL 6 =====
# ============================================================
# BIDIRECTIONAL CO-ATTENTION MODULE
# ============================================================

# Computes bidirectional cross-attention between text and image features.
# Uses projected dot-product to compute attention matrix A,
# then produces h_c = Concat(h_t @ A, h_v @ A^T) projected to d_proj.
#
# Input:  h_t [B, d_proj], h_v [B, d_proj]
# Output: h_c [B, d_proj] (co-attended representation)
class BiCoAttention(nn.Module):

    def __init__(self, d_proj=512, dropout=0.3):
        super().__init__()
        self.d_proj = d_proj

        # Projection matrix W for attention score: A = softmax(h_t W h_v^T / sqrt(d))
        self.W_attn = nn.Linear(d_proj, d_proj, bias=False)

        # Project concatenated representation [2*d_proj] back to [d_proj]
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj),
            nn.Dropout(dropout)
        )

    def forward(self, h_t, h_v):
        # h_t: [B, d], h_v: [B, d]
        # Add seq dimension for matrix ops: [B, 1, d]
        h_t_u = h_t.unsqueeze(1)  # [B, 1, d]
        h_v_u = h_v.unsqueeze(1)  # [B, 1, d]

        # Attention scores: A = softmax(h_t W h_v^T / sqrt(d))
        # h_t_u @ W: [B, 1, d], h_v_u^T: [B, d, 1]
        projected = self.W_attn(h_t_u)  # [B, 1, d]
        attn_score = torch.bmm(projected, h_v_u.transpose(1, 2)) / (self.d_proj ** 0.5)  # [B, 1, 1]
        A = torch.sigmoid(attn_score)  # [B, 1, 1] - gate-style attention for pooled features

        # Cross-attended representations
        h_t_attended = A.squeeze(-1) * h_v  # [B, d] - text-guided visual
        h_v_attended = A.squeeze(1) * h_t    # [B, d] - visual-guided text

        # Concatenate and project
        h_c = torch.cat([h_t_attended, h_v_attended], dim=1)  # [B, 2*d]
        h_c = self.fusion_proj(h_c)  # [B, d]

        return h_c


print("\u2705 BiCoAttention module defined.")

===== CELL 7 =====
# ============================================================
# ENN HEAD & SUBJECTIVE LOGIC UTILITIES
# ============================================================

# ENN Head: maps features to evidence -> Dirichlet alpha
# Input:  h [B, d_proj]
# Output: alpha [B, num_classes]
class ENNHead(nn.Module):

    def __init__(self, d_proj=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj // 2, num_classes),
            nn.Softplus()  # Non-negative evidence
        )

    def forward(self, h):
        evidence = self.fc(h)       # [B, num_classes], e >= 0
        alpha = evidence + 1        # Dirichlet parameter, alpha >= 1
        return alpha


# Subjective Logic: compute belief mass, uncertainty mass from alpha
def compute_belief_uncertainty(alpha, num_classes=3, eps=1e-8):
    # alpha: [B, K]
    S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]
    e = alpha - 1                              # [B, K]
    b = e / (S + eps)                          # belief mass [B, K]
    u = num_classes / (S + eps)                # uncertainty mass [B, 1]
    return b, u, S


print("\u2705 ENNHead & Subjective Logic utilities defined.")

===== CELL 8 =====
# ============================================================
# ADAPTIVE EVIDENTIAL FUSION (ADEF) MODULE
# ============================================================

# ADEF performs dynamic routing based on conflict level K_tv between
# text and image modalities.
#
# Route A (K_tv <= tau): Normal Dempster-Shafer fusion in 2 stages
#   Stage 1: fuse text + image -> (b_tv, u_tv)
#   Stage 2: fuse (b_tv, u_tv) + co-attention -> final
#
# Route B (K_tv > tau): Conflict-aware bypass
#   Uses K_tv as dynamic lever to blend unimodal average with co-attention
class ADEFModule(nn.Module):

    def __init__(self, num_classes=3, tau=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    # Dempster's Rule of Combination for two belief structures
    # b1, u1: belief [B, K] and uncertainty [B, 1] from source 1
    # b2, u2: belief [B, K] and uncertainty [B, 1] from source 2
    # Returns: b_fused [B, K], u_fused [B, 1]
    def dempster_combine(self, b1, u1, b2, u2):
        eps = 1e-8
        K = self.num_classes

        # Conflict: C = sum_{i!=j} b1_i * b2_j = (sum b1)(sum b2) - sum(b1*b2)
        b1_sum = torch.sum(b1, dim=1, keepdim=True)
        b2_sum = torch.sum(b2, dim=1, keepdim=True)
        C = b1_sum * b2_sum - torch.sum(b1 * b2, dim=1, keepdim=True)

        # Normalization: 1 / (1 - C)
        norm = 1.0 / (1.0 - C + eps)

        # Fused belief and uncertainty
        b_fused = norm * (b1 * b2 + b1 * u2 + b2 * u1)  # [B, K]
        u_fused = norm * (u1 * u2)                         # [B, 1]

        return b_fused, u_fused, C

    def forward(self, b_t, u_t, b_v, u_v, b_c, u_c):
        # b_t, b_v, b_c: [B, K] - belief masses
        # u_t, u_v, u_c: [B, 1] - uncertainty masses
        eps = 1e-8
        K = self.num_classes
        B = b_t.shape[0]

        # ---- Compute conflict K_tv between text and image ----
        # K_tv = sum_{i!=j} b_t_i * b_v_j
        b_t_sum = torch.sum(b_t, dim=1, keepdim=True)
        b_v_sum = torch.sum(b_v, dim=1, keepdim=True)
        K_tv = b_t_sum * b_v_sum - torch.sum(b_t * b_v, dim=1, keepdim=True)  # [B, 1]

        # ---- Dynamic Routing ----
        # Per-sample routing mask
        route_a_mask = (K_tv <= self.tau).float()  # [B, 1], 1 for Route A, 0 for Route B

        # ---- ROUTE A: Normal Dempster-Shafer Fusion (2-stage) ----
        # Stage 1: Fuse text + image
        b_tv_a, u_tv_a, _ = self.dempster_combine(b_t, u_t, b_v, u_v)

        # Stage 2: Fuse (text+image) + co-attention
        b_final_a, u_final_a, _ = self.dempster_combine(b_tv_a, u_tv_a, b_c, u_c)

        # ---- ROUTE B: Conflict-Aware Bypass Fusion ----
        # b_fusion = (1 - K_tv) * avg(b_t, b_v) + K_tv * b_c
        b_avg = (b_t + b_v) / 2.0
        b_final_b = (1.0 - K_tv) * b_avg + K_tv * b_c          # [B, K]
        u_final_b = 1.0 - torch.sum(b_final_b, dim=1, keepdim=True)  # [B, 1]
        u_final_b = torch.clamp(u_final_b, min=0.0)             # ensure non-negative

        # ---- Merge routes per sample ----
        b_fusion = route_a_mask * b_final_a + (1.0 - route_a_mask) * b_final_b
        u_fusion = route_a_mask * u_final_a + (1.0 - route_a_mask) * u_final_b

        # ---- Final Decision: p_i = b_i + u / M ----
        p_final = b_fusion + u_fusion / K  # [B, K]

        return p_final, b_fusion, u_fusion, K_tv


print("\u2705 ADEFModule defined.")

===== CELL 9 =====
# ============================================================
# MODEL: ADEFCoAttnNet
# ============================================================

# Adaptive Evidential Fusion with Co-Attention Network
# Architecture:
#   1. TextEncoder -> h_t [B, d]
#   2. ImageEncoder -> h_v [B, d]
#   3. BiCoAttention(h_t, h_v) -> h_c [B, d]
#   4. 3x ENNHead -> alpha_t, alpha_v, alpha_c
#   5. Subjective Logic -> belief & uncertainty per branch
#   6. ADEF -> dynamic routing -> final prediction
class ADEFCoAttnNet(nn.Module):

    def __init__(self, d_proj=512, num_classes=3, dropout=0.3, tau=0.5):
        super().__init__()
        # Feature extractors
        self.text_encoder = TextEncoder(d_bert=CFG.D_BERT, d_proj=d_proj)
        self.image_encoder = ImageEncoder(d_cnn=CFG.D_CNN, d_proj=d_proj)

        # Co-Attention module
        self.co_attention = BiCoAttention(d_proj=d_proj, dropout=dropout)

        # 3 independent ENN heads
        self.enn_text = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.enn_image = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.enn_coattn = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)

        # ADEF fusion module
        self.adef = ADEFModule(num_classes=num_classes, tau=tau)

        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, image):
        # 1. Feature extraction
        h_t = self.text_encoder(input_ids, attention_mask)  # [B, d]
        h_v = self.image_encoder(image)                     # [B, d]

        # 2. Co-Attention
        h_c = self.co_attention(h_t, h_v)  # [B, d]

        # 3. ENN Heads -> Dirichlet parameters
        alpha_t = self.enn_text(h_t)    # [B, K]
        alpha_v = self.enn_image(h_v)   # [B, K]
        alpha_c = self.enn_coattn(h_c)  # [B, K]

        # 4. Subjective Logic: belief & uncertainty
        b_t, u_t, S_t = compute_belief_uncertainty(alpha_t, self.num_classes)
        b_v, u_v, S_v = compute_belief_uncertainty(alpha_v, self.num_classes)
        b_c, u_c, S_c = compute_belief_uncertainty(alpha_c, self.num_classes)

        # 5. ADEF fusion
        p_final, b_fusion, u_fusion, K_tv = self.adef(b_t, u_t, b_v, u_v, b_c, u_c)

        return {
            "alpha_t": alpha_t,
            "alpha_v": alpha_v,
            "alpha_c": alpha_c,
            "b_t": b_t, "u_t": u_t,
            "b_v": b_v, "u_v": u_v,
            "b_c": b_c, "u_c": u_c,
            "p_final": p_final,
            "b_fusion": b_fusion,
            "u_fusion": u_fusion,
            "K_tv": K_tv
        }


model = ADEFCoAttnNet(
    d_proj=CFG.D_PROJ,
    num_classes=CFG.NUM_CLASSES,
    dropout=CFG.DROPOUT,
    tau=0.5
).to(CFG.DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\u2705 ADEFCoAttnNet model created.")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

===== CELL 10 =====
# ============================================================
# BLENDED LOSS FUNCTION
# ============================================================

# Evidential Loss: Bayes Risk (Sum of Squares) + KL Divergence Regularization
# L(alpha) = L_err(alpha) + lambda_t * L_KL(alpha)
class EvidentialLoss(nn.Module):

    def __init__(self, num_classes=3, annealing_epochs=10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs

    # Error Term (Sum of Squares Loss / Bayes Risk):
    # L_err = sum_j (y_j - p_j)^2 + p_j(1 - p_j) / (S + 1)
    def bayes_risk_loss(self, alpha, y_onehot):
        alpha = torch.clamp(alpha, min=1e-10)
        S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]
        p_hat = alpha / S                           # [B, K]

        # Sum of Squares error
        err = torch.sum((y_onehot - p_hat) ** 2, dim=1)

        # Variance term (epistemic uncertainty)
        var = torch.sum(p_hat * (1.0 - p_hat) / (S + 1.0), dim=1)

        return (err + var).mean()

    # KL Divergence regularization: KL(Dir(alpha_tilde) || Dir(1,...,1))
    # alpha_tilde = y + (1 - y) * alpha  (remove correct-class evidence)
    def kl_divergence_reg(self, alpha, y_onehot):
        alpha = torch.clamp(alpha, min=1e-10)
        K = self.num_classes

        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        alpha_tilde = torch.clamp(alpha_tilde, min=1e-10)
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
            + torch.sum(
                (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
                dim=1, keepdim=True
            )
        )
        return kl.mean()

    # Forward: L(alpha) = L_err + lambda_t * L_KL
    def forward(self, alpha, labels, epoch):
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        lambda_t = min(1.0, epoch / max(self.annealing_epochs, 1))

        loss_err = self.bayes_risk_loss(alpha, y_onehot)
        loss_kl = self.kl_divergence_reg(alpha, y_onehot)

        return loss_err + lambda_t * loss_kl


# Semantic Conflict Loss: L_con = d_PD * d_CC
# d_PD = 0.5 * (1 - u_t)(1 - u_v)
# d_CC = sum_i |p_t_i - p_v_i|
# Guides the encoder to detect emotional incongruence (sarcasm)
def semantic_conflict_loss(alpha_t, alpha_v, num_classes=3, eps=1e-8):
    S_t = torch.sum(alpha_t, dim=1, keepdim=True)
    S_v = torch.sum(alpha_v, dim=1, keepdim=True)

    p_t = alpha_t / (S_t + eps)          # [B, K]
    p_v = alpha_v / (S_v + eps)          # [B, K]

    u_t = num_classes / (S_t + eps)      # [B, 1]
    u_v = num_classes / (S_v + eps)      # [B, 1]

    # d_PD: confidence product - high when both modalities are confident
    d_PD = 0.5 * (1.0 - u_t) * (1.0 - u_v)  # [B, 1]

    # d_CC: cross-modal class disagreement
    d_CC = torch.sum(torch.abs(p_t - p_v), dim=1, keepdim=True)  # [B, 1]

    # L_con = d_PD * d_CC
    loss_con = (d_PD * d_CC).mean()
    return loss_con


print("\u2705 EvidentialLoss & SemanticConflictLoss defined.")

===== CELL 11 =====
# ============================================================
# UNCERTAINTY CALIBRATION ERROR (UCE)
# ============================================================

# Compute Expected Uncertainty Calibration Error.
# Bins predictions by uncertainty level, computes weighted
# |error_rate - mean_uncertainty| per bin.
# UCE = sum_j (|B_j|/N) * |err(B_j) - uncert(B_j)|
def compute_uce(predictions, labels, uncertainties, num_bins=10):
    predictions = np.array(predictions)
    labels = np.array(labels)
    uncertainties = np.array(uncertainties)
    uncertainties = np.clip(uncertainties, 0.0, 1.0)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    uce = 0.0
    N = len(predictions)

    if N == 0:
        return 0.0

    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties <= bin_boundaries[i + 1])
        else:
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_errors = (predictions[mask] != labels[mask]).astype(float).mean()
        bin_uncerts = uncertainties[mask].mean()
        uce += (n_bin / N) * np.abs(bin_errors - bin_uncerts)

    return uce


print("\u2705 UCE metric defined.")

===== CELL 12 =====
# ============================================================
# TRAINING & VALIDATION (ADEF Blended Loss)
# ============================================================

criterion = EvidentialLoss(num_classes=CFG.NUM_CLASSES, annealing_epochs=CFG.ANNEALING_EPOCHS)
GAMMA = 1.0  # balancing weight for semantic conflict loss

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG.LR,
    weight_decay=CFG.WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)

# ============================================================
# TRAINING LOOP
# L_overall = L_sup(alpha_t, alpha_v, alpha_c) + gamma * L_con
# ============================================================
history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "train_f1": [], "val_f1": [],
    "train_conflict": [], "val_conflict": []
}

best_val_f1 = 0.0
best_model_state = None

for epoch in range(1, CFG.EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    train_conflict_sum = 0.0
    train_preds, train_labels = [], []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(CFG.DEVICE)
        attention_mask = batch["attention_mask"].to(CFG.DEVICE)
        images = batch["image"].to(CFG.DEVICE)
        labels = batch["label"].to(CFG.DEVICE)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask, images)

        # Multi-task supervised loss: L_sup = L(alpha_t) + L(alpha_v) + L(alpha_c)
        loss_t = criterion(out["alpha_t"], labels, epoch)
        loss_v = criterion(out["alpha_v"], labels, epoch)
        loss_c = criterion(out["alpha_c"], labels, epoch)
        L_sup = loss_t + loss_v + loss_c

        # Semantic conflict loss
        L_con = semantic_conflict_loss(out["alpha_t"], out["alpha_v"], CFG.NUM_CLASSES)

        # Total loss
        loss = L_sup + GAMMA * L_con

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_conflict_sum += out["K_tv"].mean().item()

        # Predictions from final probability
        preds = torch.argmax(out["p_final"], dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "K_tv": f"{out['K_tv'].mean().item():.3f}"})

    scheduler.step()

    # ---- VALIDATE ----
    model.eval()
    val_loss = 0.0
    val_conflict_sum = 0.0
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS} [Val]"):
            input_ids = batch["input_ids"].to(CFG.DEVICE)
            attention_mask = batch["attention_mask"].to(CFG.DEVICE)
            images = batch["image"].to(CFG.DEVICE)
            labels = batch["label"].to(CFG.DEVICE)

            out = model(input_ids, attention_mask, images)

            loss_t = criterion(out["alpha_t"], labels, epoch)
            loss_v = criterion(out["alpha_v"], labels, epoch)
            loss_c = criterion(out["alpha_c"], labels, epoch)
            L_sup = loss_t + loss_v + loss_c
            L_con = semantic_conflict_loss(out["alpha_t"], out["alpha_v"], CFG.NUM_CLASSES)
            loss = L_sup + GAMMA * L_con

            val_loss += loss.item()
            val_conflict_sum += out["K_tv"].mean().item()

            preds = torch.argmax(out["p_final"], dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    # Epoch metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_train_conflict = train_conflict_sum / len(train_loader)
    avg_val_conflict = val_conflict_sum / len(val_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels_list, val_preds)
    train_f1 = f1_score(train_labels, train_preds, average="weighted")
    val_f1 = f1_score(val_labels_list, val_preds, average="weighted")

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)
    history["train_conflict"].append(avg_train_conflict)
    history["val_conflict"].append(avg_val_conflict)

    print(f"\nEpoch {epoch}/{CFG.EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | K_tv: {avg_train_conflict:.4f}")
    print(f"  Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | K_tv: {avg_val_conflict:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  \u2705 New best model! (Val F1: {best_val_f1:.4f})")

print(f"\n{'='*60}")
print(f"Training complete! Best Val F1: {best_val_f1:.4f}")
print(f"{'='*60}")

===== CELL 13 =====
# ============================================================
# TEST SET EVALUATION
# ============================================================

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("\u2705 Best model loaded for evaluation.")

model.eval()
all_preds = []
all_labels = []
all_uncertainties = []
all_conflicts = []
all_route_a_pct = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        input_ids = batch["input_ids"].to(CFG.DEVICE)
        attention_mask = batch["attention_mask"].to(CFG.DEVICE)
        images = batch["image"].to(CFG.DEVICE)
        labels = batch["label"].to(CFG.DEVICE)

        out = model(input_ids, attention_mask, images)

        preds = torch.argmax(out["p_final"], dim=1)
        uncertainty = out["u_fusion"].squeeze(1)  # [B]
        conflict = out["K_tv"].squeeze(1)          # [B]
        route_a = (conflict <= 0.5).float()        # [B]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_uncertainties.extend(uncertainty.cpu().numpy())
        all_conflicts.extend(conflict.cpu().numpy())
        all_route_a_pct.extend(route_a.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_uncertainties = np.array(all_uncertainties)
all_conflicts = np.array(all_conflicts)

# ============================================================
# METRICS
# ============================================================
test_acc = accuracy_score(all_labels, all_preds)
test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")
test_f1_macro = f1_score(all_labels, all_preds, average="macro")
uce_score = compute_uce(all_preds, all_labels, all_uncertainties)

print(f"\n{'='*60}")
print(f"TEST SET RESULTS (ADEF Co-Attention)")
print(f"{'='*60}")
print(f"Accuracy:          {test_acc:.4f}")
print(f"F1 (Weighted):     {test_f1_weighted:.4f}")
print(f"F1 (Macro):        {test_f1_macro:.4f}")
print(f"UCE:               {uce_score:.4f}")
print(f"{'='*60}")

print(f"\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=["Negative", "Neutral", "Positive"]
))

# ADEF routing statistics
route_a_total = np.mean(all_route_a_pct) * 100
print(f"\n{'='*60}")
print(f"ADEF ROUTING STATISTICS")
print(f"{'='*60}")
print(f"Route A (Normal DS Fusion):     {route_a_total:.1f}% of samples")
print(f"Route B (Conflict-Aware):       {100 - route_a_total:.1f}% of samples")
print(f"Mean conflict K_tv:             {all_conflicts.mean():.4f} +/- {all_conflicts.std():.4f}")

# Uncertainty analysis
correct_mask = all_preds == all_labels
print(f"\n{'='*60}")
print(f"UNCERTAINTY ANALYSIS")
print(f"{'='*60}")
if correct_mask.sum() > 0:
    print(f"Mean uncertainty (correct):   {all_uncertainties[correct_mask].mean():.4f} +/- {all_uncertainties[correct_mask].std():.4f}")
if (~correct_mask).sum() > 0:
    print(f"Mean uncertainty (incorrect): {all_uncertainties[~correct_mask].mean():.4f} +/- {all_uncertainties[~correct_mask].std():.4f}")
print(f"{'='*60}")

===== CELL 14 =====
# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle("ADEF Co-Attention + EDL - Training & Evaluation Results", fontsize=16, fontweight="bold")

# 1. Loss Curve
axes[0, 0].plot(history["train_loss"], label="Train", marker="o", markersize=3)
axes[0, 0].plot(history["val_loss"], label="Val", marker="s", markersize=3)
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss Curve"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy Curve
axes[0, 1].plot(history["train_acc"], label="Train", marker="o", markersize=3)
axes[0, 1].plot(history["val_acc"], label="Val", marker="s", markersize=3)
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Accuracy Curve"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# 3. F1 Curve
axes[0, 2].plot(history["train_f1"], label="Train", marker="o", markersize=3)
axes[0, 2].plot(history["val_f1"], label="Val", marker="s", markersize=3)
axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("F1 Score")
axes[0, 2].set_title("F1 Score Curve"); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

# 4. Conflict K_tv Curve
axes[0, 3].plot(history["train_conflict"], label="Train K_tv", marker="o", markersize=3, color="orange")
axes[0, 3].plot(history["val_conflict"], label="Val K_tv", marker="s", markersize=3, color="red")
axes[0, 3].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold (tau)")
axes[0, 3].set_xlabel("Epoch"); axes[0, 3].set_ylabel("K_tv")
axes[0, 3].set_title("Conflict Level (K_tv)"); axes[0, 3].legend(); axes[0, 3].grid(True, alpha=0.3)

# 5. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"],
            ax=axes[1, 0])
axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Actual")
axes[1, 0].set_title("Confusion Matrix")

# 6. Uncertainty Distribution (Correct vs Incorrect)
correct_mask = all_preds == all_labels
axes[1, 1].hist(all_uncertainties[correct_mask], bins=30, alpha=0.6,
                label=f"Correct (n={correct_mask.sum()})", color="green", density=True)
if (~correct_mask).sum() > 0:
    axes[1, 1].hist(all_uncertainties[~correct_mask], bins=30, alpha=0.6,
                    label=f"Incorrect (n={(~correct_mask).sum()})", color="red", density=True)
axes[1, 1].set_xlabel("Uncertainty (u)"); axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Uncertainty Distribution"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

# 7. Per-Class F1 Scores
f1_per_class = f1_score(all_labels, all_preds, average=None)
class_names = ["Negative", "Neutral", "Positive"]
bars = axes[1, 2].bar(class_names, f1_per_class, color=["#e74c3c", "#3498db", "#2ecc71"])
axes[1, 2].set_xlabel("Class"); axes[1, 2].set_ylabel("F1 Score")
axes[1, 2].set_title("Per-Class F1 Score"); axes[1, 2].set_ylim(0, 1)
for bar, val in zip(bars, f1_per_class):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
axes[1, 2].grid(True, alpha=0.3, axis="y")

# 8. Conflict Distribution & Route Breakdown
axes[1, 3].hist(all_conflicts[all_conflicts <= 0.5], bins=20, alpha=0.6,
                label=f"Route A (n={(all_conflicts <= 0.5).sum()})", color="blue", density=True)
axes[1, 3].hist(all_conflicts[all_conflicts > 0.5], bins=20, alpha=0.6,
                label=f"Route B (n={(all_conflicts > 0.5).sum()})", color="orange", density=True)
axes[1, 3].axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="tau=0.5")
axes[1, 3].set_xlabel("Conflict (K_tv)"); axes[1, 3].set_ylabel("Density")
axes[1, 3].set_title("ADEF Routing Distribution"); axes[1, 3].legend(); axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\u2705 Visualization complete.")


```

---

## Assistant (Plan · Kimi K3 (2x usage) · 10.5s)

_Thinking:_

Now let me look at the outputs of the cells, especially cell 12 (training, 61 outputs) and cell 13 (test results) to see the neutral class issue.

Now let me check the training outputs and test results to see the neutral class problem:

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | ConvertFrom-Json; $cell = $nb.cells[13]; foreach ($o in $cell.outputs) { if ($o.output_type -eq 'stream') { ($o.text -join '') } elseif ($o.output_type -eq 'execute_result' -and $o.data.'text/plain') { ($o.data.'text/plain' -join '') } }"
}
```

**Output:**
```
? Best model loaded for evaluation.

Evaluating on Test Set: 100%|����������| 43/43 [00:16<00:00,  2.61it/s]

============================================================
TEST SET RESULTS (ADEF Co-Attention)
============================================================
Accuracy:          0.6721
F1 (Weighted):     0.6201
F1 (Macro):        0.4324
UCE:               0.0586
============================================================

Classification Report:
              precision    recall  f1-score   support

    Negative       0.62      0.44      0.52       204
     Neutral       0.00      0.00      0.00        70
    Positive       0.68      0.91      0.78       403

    accuracy                           0.67       677
   macro avg       0.44      0.45      0.43       677
weighted avg       0.60      0.67      0.62       677


============================================================
ADEF ROUTING STATISTICS
============================================================
Route A (Normal DS Fusion):     100.0% of samples
Route B (Conflict-Aware):       0.0% of samples
Mean conflict K_tv:             0.0140 +/- 0.0094

============================================================
UNCERTAINTY ANALYSIS
============================================================
Mean uncertainty (correct):   0.3056 +/- 0.1709
Mean uncertainty (incorrect): 0.4115 +/- 0.1650
============================================================




```

---

## Assistant (Plan · Kimi K3 (2x usage) · 15.9s)

_Thinking:_

This is very revealing. The test results show:

- **Neutral class: precision 0.00, recall 0.00, F1 0.00** — the model NEVER predicts neutral!
- Route A is used 100% of the time; Route B never triggers (mean K_tv = 0.014, way below tau=0.5)
- Accuracy 67.2%, but macro F1 is only 0.43

So the issue: the model never predicts neutral. Why?

Let me check the training outputs to understand class distribution and training dynamics.

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | ConvertFrom-Json; $cell = $nb.cells[12]; foreach ($o in $cell.outputs) { if ($o.output_type -eq 'stream') { ($o.text -join '') } }"
}
```

**Output:**
```
Epoch 1/30 [Train]: 100%|����������| 198/198 [01:23<00:00,  2.36it/s, loss=2.8376, K_tv=0.100]
Epoch 1/30 [Val]: 100%|����������| 43/43 [00:16<00:00,  2.63it/s]


Epoch 1/30
  Train Loss: 2.1160 | Acc: 0.5869 | F1: 0.4577 | K_tv: 0.1075
  Val   Loss: 2.0274 | Acc: 0.5938 | F1: 0.4425 | K_tv: 0.1050
  LR: 1.99e-05
  ? New best model! (Val F1: 0.4425)

Epoch 2/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.52it/s, loss=1.2837, K_tv=0.063]
Epoch 2/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.99it/s]


Epoch 2/30
  Train Loss: 2.0789 | Acc: 0.5949 | F1: 0.4438 | K_tv: 0.0803
  Val   Loss: 2.0951 | Acc: 0.5938 | F1: 0.4425 | K_tv: 0.0681
  LR: 1.98e-05

Epoch 3/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.53it/s, loss=1.7136, K_tv=0.046]
Epoch 3/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.99it/s]


Epoch 3/30
  Train Loss: 2.0973 | Acc: 0.5961 | F1: 0.4467 | K_tv: 0.0544
  Val   Loss: 2.1373 | Acc: 0.5953 | F1: 0.4458 | K_tv: 0.0500
  LR: 1.95e-05
  ? New best model! (Val F1: 0.4458)

Epoch 4/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.53it/s, loss=2.1093, K_tv=0.040]
Epoch 4/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.98it/s]


Epoch 4/30
  Train Loss: 2.1134 | Acc: 0.6044 | F1: 0.4673 | K_tv: 0.0408
  Val   Loss: 2.1607 | Acc: 0.6041 | F1: 0.4793 | K_tv: 0.0380
  LR: 1.91e-05
  ? New best model! (Val F1: 0.4793)

Epoch 5/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.49it/s, loss=2.3321, K_tv=0.030]
Epoch 5/30 [Val]: 100%|����������| 43/43 [00:11<00:00,  3.88it/s]


Epoch 5/30
  Train Loss: 2.1230 | Acc: 0.6231 | F1: 0.5116 | K_tv: 0.0320
  Val   Loss: 2.1911 | Acc: 0.6115 | F1: 0.5121 | K_tv: 0.0299
  LR: 1.87e-05
  ? New best model! (Val F1: 0.5121)

Epoch 6/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.50it/s, loss=1.4112, K_tv=0.021]
Epoch 6/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.96it/s]


Epoch 6/30
  Train Loss: 2.1207 | Acc: 0.6430 | F1: 0.5539 | K_tv: 0.0260
  Val   Loss: 2.2283 | Acc: 0.6233 | F1: 0.5401 | K_tv: 0.0259
  LR: 1.81e-05
  ? New best model! (Val F1: 0.5401)

Epoch 7/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.51it/s, loss=1.7258, K_tv=0.019]
Epoch 7/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.00it/s]


Epoch 7/30
  Train Loss: 2.1143 | Acc: 0.6595 | F1: 0.5855 | K_tv: 0.0221
  Val   Loss: 2.2659 | Acc: 0.6174 | F1: 0.5210 | K_tv: 0.0220
  LR: 1.74e-05

Epoch 8/30 [Train]: 100%|����������| 198/198 [01:12<00:00,  2.73it/s, loss=2.3575, K_tv=0.015]
Epoch 8/30 [Val]: 100%|����������| 43/43 [00:15<00:00,  2.83it/s]


Epoch 8/30
  Train Loss: 2.1177 | Acc: 0.6668 | F1: 0.5960 | K_tv: 0.0196
  Val   Loss: 2.2795 | Acc: 0.6440 | F1: 0.5758 | K_tv: 0.0194
  LR: 1.67e-05
  ? New best model! (Val F1: 0.5758)

Epoch 9/30 [Train]: 100%|����������| 198/198 [01:14<00:00,  2.64it/s, loss=2.3809, K_tv=0.015]
Epoch 9/30 [Val]: 100%|����������| 43/43 [00:14<00:00,  2.88it/s]


Epoch 9/30
  Train Loss: 2.1203 | Acc: 0.6718 | F1: 0.6034 | K_tv: 0.0171
  Val   Loss: 2.2978 | Acc: 0.6352 | F1: 0.5612 | K_tv: 0.0171
  LR: 1.59e-05

Epoch 10/30 [Train]: 100%|����������| 198/198 [01:15<00:00,  2.62it/s, loss=2.3830, K_tv=0.017]
Epoch 10/30 [Val]: 100%|����������| 43/43 [00:15<00:00,  2.86it/s]


Epoch 10/30
  Train Loss: 2.1069 | Acc: 0.6820 | F1: 0.6192 | K_tv: 0.0148
  Val   Loss: 2.3444 | Acc: 0.6322 | F1: 0.5600 | K_tv: 0.0145
  LR: 1.50e-05

Epoch 11/30 [Train]: 100%|����������| 198/198 [01:15<00:00,  2.64it/s, loss=1.7887, K_tv=0.011]
Epoch 11/30 [Val]: 100%|����������| 43/43 [00:14<00:00,  2.88it/s]


Epoch 11/30
  Train Loss: 2.0882 | Acc: 0.6832 | F1: 0.6230 | K_tv: 0.0142
  Val   Loss: 2.3461 | Acc: 0.6470 | F1: 0.5857 | K_tv: 0.0140
  LR: 1.41e-05
  ? New best model! (Val F1: 0.5857)

Epoch 12/30 [Train]: 100%|����������| 198/198 [01:14<00:00,  2.66it/s, loss=1.8132, K_tv=0.017]
Epoch 12/30 [Val]: 100%|����������| 43/43 [00:14<00:00,  2.89it/s]


Epoch 12/30
  Train Loss: 2.0727 | Acc: 0.6988 | F1: 0.6417 | K_tv: 0.0141
  Val   Loss: 2.3169 | Acc: 0.6603 | F1: 0.6029 | K_tv: 0.0134
  LR: 1.31e-05
  ? New best model! (Val F1: 0.6029)

Epoch 13/30 [Train]: 100%|����������| 198/198 [01:15<00:00,  2.63it/s, loss=3.4806, K_tv=0.009]
Epoch 13/30 [Val]: 100%|����������| 43/43 [00:14<00:00,  2.87it/s]


Epoch 13/30
  Train Loss: 2.0579 | Acc: 0.7067 | F1: 0.6492 | K_tv: 0.0139
  Val   Loss: 2.3170 | Acc: 0.6544 | F1: 0.6038 | K_tv: 0.0143
  LR: 1.21e-05
  ? New best model! (Val F1: 0.6038)

Epoch 14/30 [Train]: 100%|����������| 198/198 [01:15<00:00,  2.64it/s, loss=1.9517, K_tv=0.014]
Epoch 14/30 [Val]: 100%|����������| 43/43 [00:14<00:00,  3.04it/s]


Epoch 14/30
  Train Loss: 2.0228 | Acc: 0.7086 | F1: 0.6535 | K_tv: 0.0139
  Val   Loss: 2.3185 | Acc: 0.6499 | F1: 0.5989 | K_tv: 0.0154
  LR: 1.10e-05

Epoch 15/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.48it/s, loss=1.9515, K_tv=0.013]
Epoch 15/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.99it/s]


Epoch 15/30
  Train Loss: 1.9967 | Acc: 0.7124 | F1: 0.6574 | K_tv: 0.0139
  Val   Loss: 2.3325 | Acc: 0.6484 | F1: 0.5972 | K_tv: 0.0134
  LR: 1.00e-05

Epoch 16/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.54it/s, loss=1.9864, K_tv=0.011]
Epoch 16/30 [Val]: 100%|����������| 43/43 [00:11<00:00,  3.89it/s]


Epoch 16/30
  Train Loss: 1.9936 | Acc: 0.7194 | F1: 0.6679 | K_tv: 0.0135
  Val   Loss: 2.3324 | Acc: 0.6573 | F1: 0.6032 | K_tv: 0.0147
  LR: 8.95e-06

Epoch 17/30 [Train]: 100%|����������| 198/198 [00:52<00:00,  3.76it/s, loss=1.9987, K_tv=0.011]
Epoch 17/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.28it/s]


Epoch 17/30
  Train Loss: 1.9763 | Acc: 0.7247 | F1: 0.6724 | K_tv: 0.0138
  Val   Loss: 2.3262 | Acc: 0.6529 | F1: 0.6025 | K_tv: 0.0154
  LR: 7.92e-06

Epoch 18/30 [Train]: 100%|����������| 198/198 [00:51<00:00,  3.83it/s, loss=1.8921, K_tv=0.017]
Epoch 18/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.24it/s]


Epoch 18/30
  Train Loss: 1.9641 | Acc: 0.7304 | F1: 0.6789 | K_tv: 0.0144
  Val   Loss: 2.3294 | Acc: 0.6425 | F1: 0.5957 | K_tv: 0.0154
  LR: 6.91e-06

Epoch 19/30 [Train]: 100%|����������| 198/198 [00:51<00:00,  3.82it/s, loss=2.5298, K_tv=0.018]
Epoch 19/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.23it/s]


Epoch 19/30
  Train Loss: 1.9466 | Acc: 0.7396 | F1: 0.6889 | K_tv: 0.0141
  Val   Loss: 2.3404 | Acc: 0.6499 | F1: 0.5990 | K_tv: 0.0142
  LR: 5.93e-06

Epoch 20/30 [Train]: 100%|����������| 198/198 [00:51<00:00,  3.82it/s, loss=2.3430, K_tv=0.011]
Epoch 20/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.23it/s]


Epoch 20/30
  Train Loss: 1.9453 | Acc: 0.7377 | F1: 0.6867 | K_tv: 0.0140
  Val   Loss: 2.3171 | Acc: 0.6484 | F1: 0.6014 | K_tv: 0.0139
  LR: 5.00e-06

Epoch 21/30 [Train]: 100%|����������| 198/198 [00:52<00:00,  3.80it/s, loss=3.1914, K_tv=0.015]
Epoch 21/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.26it/s]


Epoch 21/30
  Train Loss: 1.9298 | Acc: 0.7466 | F1: 0.6971 | K_tv: 0.0139
  Val   Loss: 2.3370 | Acc: 0.6529 | F1: 0.6069 | K_tv: 0.0152
  LR: 4.12e-06
  ? New best model! (Val F1: 0.6069)

Epoch 22/30 [Train]: 100%|����������| 198/198 [00:52<00:00,  3.80it/s, loss=2.1640, K_tv=0.011]
Epoch 22/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.26it/s]


Epoch 22/30
  Train Loss: 1.9310 | Acc: 0.7396 | F1: 0.6877 | K_tv: 0.0143
  Val   Loss: 2.3375 | Acc: 0.6647 | F1: 0.6177 | K_tv: 0.0146
  LR: 3.31e-06
  ? New best model! (Val F1: 0.6177)

Epoch 23/30 [Train]: 100%|����������| 198/198 [00:52<00:00,  3.79it/s, loss=1.6272, K_tv=0.010]
Epoch 23/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.27it/s]


Epoch 23/30
  Train Loss: 1.9126 | Acc: 0.7485 | F1: 0.6990 | K_tv: 0.0140
  Val   Loss: 2.3569 | Acc: 0.6617 | F1: 0.6107 | K_tv: 0.0150
  LR: 2.57e-06

Epoch 24/30 [Train]: 100%|����������| 198/198 [00:52<00:00,  3.79it/s, loss=2.3412, K_tv=0.011]
Epoch 24/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.22it/s]


Epoch 24/30
  Train Loss: 1.9137 | Acc: 0.7422 | F1: 0.6912 | K_tv: 0.0143
  Val   Loss: 2.3533 | Acc: 0.6573 | F1: 0.6106 | K_tv: 0.0153
  LR: 1.91e-06

Epoch 25/30 [Train]: 100%|����������| 198/198 [00:53<00:00,  3.72it/s, loss=2.8836, K_tv=0.009]
Epoch 25/30 [Val]: 100%|����������| 43/43 [00:11<00:00,  3.90it/s]


Epoch 25/30
  Train Loss: 1.9144 | Acc: 0.7589 | F1: 0.7106 | K_tv: 0.0143
  Val   Loss: 2.3463 | Acc: 0.6573 | F1: 0.6125 | K_tv: 0.0151
  LR: 1.34e-06

Epoch 26/30 [Train]: 100%|����������| 198/198 [00:55<00:00,  3.58it/s, loss=1.8851, K_tv=0.018]
Epoch 26/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.04it/s]


Epoch 26/30
  Train Loss: 1.9069 | Acc: 0.7542 | F1: 0.7040 | K_tv: 0.0144
  Val   Loss: 2.3372 | Acc: 0.6691 | F1: 0.6213 | K_tv: 0.0152
  LR: 8.65e-07
  ? New best model! (Val F1: 0.6213)

Epoch 27/30 [Train]: 100%|����������| 198/198 [00:55<00:00,  3.56it/s, loss=1.8095, K_tv=0.011]
Epoch 27/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.03it/s]


Epoch 27/30
  Train Loss: 1.8968 | Acc: 0.7593 | F1: 0.7103 | K_tv: 0.0143
  Val   Loss: 2.3534 | Acc: 0.6603 | F1: 0.6134 | K_tv: 0.0154
  LR: 4.89e-07

Epoch 28/30 [Train]: 100%|����������| 198/198 [00:55<00:00,  3.57it/s, loss=2.0159, K_tv=0.013]
Epoch 28/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.94it/s]


Epoch 28/30
  Train Loss: 1.9094 | Acc: 0.7542 | F1: 0.7052 | K_tv: 0.0145
  Val   Loss: 2.3522 | Acc: 0.6588 | F1: 0.6116 | K_tv: 0.0154
  LR: 2.19e-07

Epoch 29/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.50it/s, loss=1.7904, K_tv=0.017]
Epoch 29/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  3.96it/s]


Epoch 29/30
  Train Loss: 1.8919 | Acc: 0.7510 | F1: 0.7020 | K_tv: 0.0145
  Val   Loss: 2.3529 | Acc: 0.6662 | F1: 0.6182 | K_tv: 0.0148
  LR: 5.48e-08

Epoch 30/30 [Train]: 100%|����������| 198/198 [00:56<00:00,  3.48it/s, loss=2.3965, K_tv=0.012]
Epoch 30/30 [Val]: 100%|����������| 43/43 [00:10<00:00,  4.01it/s]

Epoch 30/30
  Train Loss: 1.8934 | Acc: 0.7589 | F1: 0.7105 | K_tv: 0.0144
  Val   Loss: 2.3545 | Acc: 0.6632 | F1: 0.6163 | K_tv: 0.0152
  LR: 0.00e+00

============================================================
Training complete! Best Val F1: 0.6213
============================================================




```

---

## Assistant (Plan · Kimi K3 (2x usage) · 10.0s)

_Thinking:_

Now let me check cell 3 and 4 outputs for the class distribution:

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | ConvertFrom-Json; foreach ($idx in @(3,4)) { $cell = $nb.cells[$idx]; \"===== CELL $idx OUTPUTS =====\"; foreach ($o in $cell.outputs) { if ($o.output_type -eq 'stream') { ($o.text -join '') } elseif ($o.data -and $o.data.'text/plain') { ($o.data.'text/plain' -join '') } } }"
}
```

**Output:**
```
===== CELL 3 OUTPUTS =====
Dataset size after filtering: 4511

============================================================
PREPROCESSING STATISTICS:
============================================================
Total samples: 4511
Samples with EMPTY text: 0
Samples with VALID text: 4511
Percentage of empty text: 0.00%
============================================================


   id text_label image_label final_label  label  \
0   1    neutral    positive    positive      2   
1   2    neutral    positive    positive      2   
2   3    neutral    positive    positive      2   
3   4   positive    positive    positive      2   
4   5   positive    positive    positive      2   

                                                text  \
0       How I feel today #legday #jelly #aching #gym   
1  grattis min griskulting!!!???? va bara tvungen...   
2  RT @polynminion: The moment I found my favouri...   
3  #escort We have a young and energetic team and...   
4  RT @chrisashaffer: Went to SSC today to be a "...   

                  image_path  
0  D:/MVSA_SINGLE/data\1.jpg  
1  D:/MVSA_SINGLE/data\2.jpg  
2  D:/MVSA_SINGLE/data\3.jpg  
3  D:/MVSA_SINGLE/data\4.jpg  
4  D:/MVSA_SINGLE/data\5.jpg  
===== CELL 4 OUTPUTS =====
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Train: 3157 | Val: 677 | Test: 677

Train label distribution:
label
0     950
1     329
2    1878
Name: count, dtype: int64

Val label distribution:
label
0    204
1     71
2    402
Name: count, dtype: int64

Test label distribution:
label
0    204
1     70
2    403
Name: count, dtype: int64

? DataLoaders created.
Train batches: 198 | Val batches: 43 | Test batches: 43


```

---

## Assistant (Plan · Kimi K3 (2x usage) · 86.8s)

_Thinking:_

Now I have the full picture. Let me analyze the root causes of the neutral-class failure:

**Class distribution (severe imbalance):**
- Train: Negative 950 (30%), Neutral 329 (10.4%), Positive 1878 (59.5%)
- Neutral is only ~10% of the data!

**Test results:**
- Neutral: precision 0, recall 0, F1 0 — the model NEVER predicts neutral
- Negative: recall 0.44
- Positive: recall 0.91 (dominant class wins)

**Root causes of neutral misprediction:**

1. **Severe class imbalance (10% neutral)** with no handling: no class weights, no balanced sampling, no focal loss. The Bayes-risk/SoS evidential loss + argmax decision is dominated by the majority class.

2. **Both encoders are FROZEN** (`requires_grad = False`, `torch.no_grad()`). Only small projection layers + ENN heads are trained (let me estimate trainable params). RoBERTa-base has 125M params, DenseNet121 has 8M — all frozen. Only ~1.5-2M trainable params in projections/heads. The features are generic ImageNet/MLM features, not sentiment-discriminative. Neutral class is semantically subtle ("absence of sentiment") — frozen generic features + tiny heads can't carve out a neutral region.

3. **Co-attention is broken/degenerate**: The "co-attention" operates on POOLED vectors [B, d] not sequences. The attention matrix A is a single scalar [B,1,1] computed from two pooled vectors, passed through sigmoid. So h_c is just element-wise scaling: A*h_v and A*h_t. That's not co-attention at all — it's a scalar gate. The thesis plan describes token-level/region-level co-attention with A = softmax(h_t W h_v^T / sqrt(d)) where h_t is [B, L_t, d] and h_v is [B, N_v, d]. The current implementation pools first, so there's no fine-grained alignment. Also the plan says softmax, but implementation uses sigmoid on a [B,1,1] logit.

4. **Route B never fires**: K_tv mean = 0.014, so tau=0.5 means Route B is used 0.0% of the time. The whole "adaptive" mechanism is dead code. Why is K_tv so tiny? Because K_tv = sum_{i≠j} b_t,i * b_v,j and with 3 classes, if beliefs are diffuse or aligned, K_tv is small. Actually let's think: max conflict happens when b_t = (1,0,0) and b_v = (0,1,0) → K_tv = 1. But with trained aligned modalities, beliefs agree → K_tv ≈ small. Also since both unimodal heads are trained on the same label with the same loss, they tend to agree → low conflict by design. The threshold tau=0.5 is on a different scale than typical K_tv values (max K_tv for 3-class with uniform-belief... let's compute: if b_t=(0.5,0.5,0), b_v=(0,0.5,0.5)... K = 0.5*0.5+0.5*0+... = 0.25). Realistically K_tv rarely exceeds 0.5. So the routing threshold needs to be on the observed scale (e.g., 0.2-0.3) or use a soft/differentiable gate, or normalize K by its max possible value.

5. **Evidential loss with argmax(p_final)**: neutral is the "in-between" class; with sum-of-squares Bayes risk and a uniform-prior uncertainty distribution p_i = b_i + u/M, uncertain samples get spread uniformly — this favors boundary classes... Actually the uniform distribution of u doesn't favor neutral specifically. The bigger issue is imbalance + weak features.

6. **LR schedule**: CosineAnnealingLR with T_max=30 reaches 0 at epoch 30 — fine. But LR=2e-5 is very low for training randomly-initialized projection layers and heads (2e-5 is a fine-tuning LR for transformers, but here transformers are frozen; the heads need higher LR like 1e-3 - 1e-4). This is a key issue: they froze the backbones (no need for 2e-5) and then use a tiny LR on the small heads → undertraining. Train acc only reaches 75.9% after 30 epochs while val ~66%.

7. **No label smoothing, no balanced batch sampler**.

8. **Dempster fusion issue**: In Route A stage 1, Dempster combination with normalization 1/(1-K). Fine. But fusing three heads where each head outputs belief close to uniform-ish... The final decision p = b + u/3.

9. **Also the ENN heads use Softplus which at init gives softplus(0)≈0.693 evidence per class → alpha ≈ 1.69 per class, u = 3/(3*1.69) ≈ 0.59. Fine.

10. **UCE is quite low (0.0586) but meaningless when the model collapses to majority.**

Another important detail: the training loop tracks metrics via p_final argmax; val F1 weighted 0.62 → the model isn't that bad on weighted but macro is terrible.

**Plan for fixing:**

A. **Root-cause explanation** (to tell the user why neutral fails):
1. Class imbalance 59/30/10 with no compensation → model collapses to majority (positive) and treats neutral as "noise". Neutral F1 = 0.
2. Frozen backbones + tiny LR → undertrained heads; only ~2M trainable params; LR 2e-5 is for fine-tuning, not training fresh heads.
3. Co-attention degenerated to a scalar gate on pooled vectors — no fine-grained alignment as in the thesis plan → weak h_c branch.
4. K_tv stays ~0.014 ≪ tau=0.5 → Route B never triggers; ADEF routing is effectively dead; "adaptive" contribution zero.
5. Neutral is inherently "absence of evidence" — in EDL, neutral should be modeled as high-uncertainty/low-evidence, but the decision p_i = b_i + u/M spreads u uniformly, which doesn't favor neutral; plus supervised heads push evidence to majority classes.

B. **Fixes to the notebook:**
1. Make ALL hyperparameters config: tau, gamma, annealing epochs, dropout, LR (separate LR for heads vs backbone if unfrozen), weight decay, batch size, epochs, d_proj, num classes, max_len, class-weight toggle, etc. — move GAMMA=1.0, tau=0.5, clip norm, T_max etc. into CFG.
2. Handle imbalance: add class weights in evidential loss (weight per-class in the SoS error term), and/or use a WeightedRandomSampler. Simplest minimal change: compute class weights from train_df and pass to EvidentialLoss.
3. Unfreeze top layers of backbones with discriminative LR (or at least higher LR for heads: param groups — heads LR 1e-4~5e-4, backbone 1e-5~2e-5). Minimal: unfreeze last N transformer layers of RoBERTa + last dense block of DenseNet.
4. Fix co-attention to be true token/patch-level: keep RoBERTa last_hidden_state [B, L, 768] projected to d, and DenseNet feature map [B, 1024, 7, 7] → flatten to 49 tokens projected to d. Then A = softmax over rows/cols of h_t W h_v^T / sqrt(d), h_c = concat of attended summaries. But that requires unfreezing and more compute. There's a tradeoff: full co-attention is heavier but matches thesis plan and is the "proposed model" contribution. Since this is the thesis model (the main contribution), it should be done right.
5. Fix routing: make tau configurable (user asked), but also consider making the route decision soft (sigmoid((K_tv - tau)/temperature)) for differentiability — though the hard mask is fine since K_tv is computed from beliefs and gradients flow through both branches' outputs anyway... Actually with hard mask, gradient flows only through the selected route — that's fine for training since loss is on alphas (supervised heads), not on fusion output. Note: the current loss does NOT supervise p_final at all! The loss is only on alpha_t, alpha_v, alpha_c + conflict loss. The fusion module receives no gradient except through... actually p_final is not in the loss, so ADEF fusion params (none — it's parameter-free) get no gradient. The fusion is parameter-free, so that's OK. But then the decision quality depends entirely on the three heads being individually good.
   - Add a supervised loss on the fused prediction too? The plan says L_sup = L(alpha_t) + L(alpha_v) + L(alpha_c). To improve final decision, we could derive a fused alpha (equivalent Dirichlet from b_fusion, u_fusion) and add its evidential loss. This is an enhancement I can propose: convert fused opinion back to alpha_fusion = b_fusion * S_f + 1 with S_f = M/u_f. Then add L(alpha_fusion) to the total. This directly optimizes the final decision, helping neutral too (with class weights).
6. Neutral-specific: with EDL, optionally add a regularizer encouraging high u when unimodal heads disagree... that's L_con already. But L_con weight gamma=1.0 — L_con values: d_PD*d_CC is small; fine.

C. **Rewrite the thesis plan (rancangan_tesis_adef_co_attention.md)**: user says "my plan is from [md file] but i don't know this is optimal or not. you can free to modify and give the best proposed model. just modify the old plan". So I should edit the md file to reflect the improved design:
- True sequence-level co-attention (token-level text, patch-level image)
- Partial fine-tuning of encoders
- Class-imbalance handling (class-weighted evidential loss)
- Fused-opinion supervision term
- Config-driven hyperparameters including tau
- Routing threshold calibration (tau as config, plus note about typical K_tv scale; optionally soft routing)
- Keep the ADEF routing idea (it's the thesis novelty), but fix scale issue: e.g., normalize K_tv or set tau based on validation K_tv distribution; add that as a recommendation.

Wait — the user wants me to "fix my adef architecture, why it can miss predicted in neutral. and also make the tau as a config, do for all hyperparameter." — so fix the notebook AND modify the plan md. But I'm in PLAN MODE. I must not edit anything. I need to produce a plan and ask clarifying questions.

Let me also check the other notebooks briefly to see how they compare (e.g., co_attention.ipynb might have a working co-attention implementation to borrow). That would help the plan. Let me quickly peek at co_attention.ipynb's model cell and check GPU. Also check the results png? Not necessary.

Let me look at co_attention.ipynb quickly for the co-attention implementation used there.

Also check what GPU is available (cell 1 output of adef notebook). Let me check cell 1 and 2 outputs.

Let me check the model parameter counts and compare with the sibling `co_attention.ipynb` implementation:

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" | ConvertFrom-Json; foreach ($idx in @(1,2,9)) { $cell = $nb.cells[$idx]; \"===== CELL $idx OUTPUTS =====\"; foreach ($o in $cell.outputs) { if ($o.output_type -eq 'stream') { ($o.text -join '') } } }"
}
```

**Output:**
```
===== CELL 1 OUTPUTS =====
C:\Users\Residensi ADW\AppData\Roaming\Python\Python310\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

? Imports loaded & seed set.
PyTorch version: 2.5.1
CUDA available: True
GPU: NVIDIA GeForce RTX 3060

===== CELL 2 OUTPUTS =====
? Configuration loaded. Device: cuda

===== CELL 9 OUTPUTS =====
Loading weights: 100%|����������| 197/197 [00:00<00:00, 5792.95it/s]
[transformers] [1mRobertaModel LOAD REPORT[0m from: roberta-base
Key                       | Status     | 
--------------------------+------------+-
lm_head.dense.bias        | UNEXPECTED | 
lm_head.bias              | UNEXPECTED | 
lm_head.dense.weight      | UNEXPECTED | 
lm_head.layer_norm.bias   | UNEXPECTED | 
lm_head.layer_norm.weight | UNEXPECTED | 
pooler.dense.bias         | MISSING    | 
pooler.dense.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

? ADEFCoAttnNet model created.
Total parameters: 133,704,329
Trainable parameters: 2,104,841
Frozen parameters: 131,599,488


```

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\co_attention.ipynb\" | ConvertFrom-Json; $i=0; foreach ($c in $nb.cells) { if ($c.cell_type -eq 'code') { $src = ($c.source -join ''); if ($src -match 'class .*Attention' -or $src -match 'class .*Net') { \"===== CELL $i =====\"; $src } }; $i++ }"
}
```

**Output:**
```
===== CELL 7 =====
# ============================================================
# MODEL: Co-Attention Fusion + EDL
# ============================================================

# Co-Attention: Bidirectional interactive attention.
#
# Architecture:
#   Path 1 - Text-guided Visual Attention: Text(Q) -> Image(K,V)
#   Path 2 - Image-guided Textual Attention: Image(Q) -> Text(K,V)
#   Adaptive Gating to fuse both attended representations
#   -> ENN Head -> Evidence -> alpha
class CoAttentionEDL(nn.Module):

    def __init__(self, d_proj=512, num_classes=3, num_heads=8, dropout=0.3):
        super().__init__()
        self.text_encoder = TextEncoder(d_proj)
        self.image_encoder = ImageEncoder(d_proj)
        self.d_proj = d_proj

        # Text-guided Visual Attention (Text -> Q, Image -> K, V)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=d_proj,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_t2i = nn.LayerNorm(d_proj)

        # Image-guided Textual Attention (Image -> Q, Text -> K, V)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_proj,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_i2t = nn.LayerNorm(d_proj)

        # Adaptive Gating Mechanism
        self.gate = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj),
            nn.Sigmoid()
        )

        # Projection after gated fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_proj, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj),
            nn.Dropout(dropout)
        )

        # ENN Head
        self.enn_head = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj // 2, num_classes),
            nn.Softplus()
        )

    def forward(self, input_ids, attention_mask, image):
        # Get sequence-level features
        text_seq, text_mask = self.text_encoder(input_ids, attention_mask)  # [B, L_t, d], [B, L_t]
        image_spatial = self.image_encoder(image)                           # [B, H*W, d]

        # ===== Path 1: Text-guided Visual Attention =====
        # Text queries attend to image spatial features
        t2i_output, _ = self.text_to_image_attn(
            query=text_seq,        # [B, L_t, d_proj]
            key=image_spatial,     # [B, H*W, d_proj]
            value=image_spatial    # [B, H*W, d_proj]
        )
        t2i_output = self.ln_t2i(text_seq + t2i_output)  # [B, L_t, d_proj]

        # Mean pool text-attended representation (with mask)
        mask_t = text_mask.unsqueeze(-1).float()  # [B, L_t, 1]
        h_t2i = (t2i_output * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1e-10)  # [B, d_proj]

        # ===== Path 2: Image-guided Textual Attention =====
        # Image queries attend to text sequence features
        text_key_padding_mask = (text_mask == 0)  # [B, L_t] True=ignore

        i2t_output, _ = self.image_to_text_attn(
            query=image_spatial,   # [B, H*W, d_proj]
            key=text_seq,          # [B, L_t, d_proj]
            value=text_seq,        # [B, L_t, d_proj]
            key_padding_mask=text_key_padding_mask
        )
        i2t_output = self.ln_i2t(image_spatial + i2t_output)  # [B, H*W, d_proj]

        # Mean pool image-attended representation
        h_i2t = i2t_output.mean(dim=1)  # [B, d_proj]

        # ===== Adaptive Gating =====
        combined = torch.cat([h_t2i, h_i2t], dim=1)  # [B, 2*d_proj]
        gate_weight = self.gate(combined)              # [B, d_proj] in [0, 1]
        fused = gate_weight * h_t2i + (1 - gate_weight) * h_i2t  # [B, d_proj]

        fused = self.fusion_proj(fused)  # [B, d_proj]

        # ENN Head
        evidence = self.enn_head(fused)  # [B, num_classes]
        alpha = evidence + 1

        return alpha


model = CoAttentionEDL(
    d_proj=CFG.D_PROJ,
    num_classes=CFG.NUM_CLASSES,
    num_heads=8,
    dropout=CFG.DROPOUT
).to(CFG.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\u2705 CoAttentionEDL model created.")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")


```

---

## Assistant (Plan · Kimi K3 (2x usage) · 15.1s)

_Thinking:_

Very useful. The sibling `co_attention.ipynb` has sequence-level encoders (TextEncoder returns [B, L_t, d] + mask, ImageEncoder returns [B, H*W, d]) and uses proper multihead cross-attention. Let me check the TextEncoder/ImageEncoder in co_attention.ipynb to see how they output sequences, and its results for comparison (does it also fail on neutral?).

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\co_attention.ipynb\" | ConvertFrom-Json; $i=0; foreach ($c in $nb.cells) { if ($c.cell_type -eq 'code') { $src = ($c.source -join ''); if ($src -match 'class TextEncoder' -or $src -match 'class ImageEncoder') { \"===== CELL $i =====\"; $src } }; $i++ }"
}
```

**Output:**
```
===== CELL 5 =====
# ============================================================
# FEATURE EXTRACTORS (Sequence-Level for Attention)
# ============================================================

# RoBERTa text encoder returning sequence-level features.
# Input:  input_ids [B, L_t], attention_mask [B, L_t]
# Output: sequence [B, L_t, d_proj], attention_mask [B, L_t]
class TextEncoder(nn.Module):

    def __init__(self, d_proj=512):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(CFG.TEXT_MODEL)
        # Freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(CFG.D_BERT, d_proj),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(d_proj)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L_t, 768]
        projected = self.layer_norm(self.projection(sequence_output))  # [B, L_t, d_proj]
        return projected, attention_mask


# DenseNet-121 image encoder returning spatial features.
# Input:  image [B, 3, 224, 224]
# Output: spatial [B, H*W, d_proj]
class ImageEncoder(nn.Module):

    def __init__(self, d_proj=512):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = densenet.features
        # Freeze DenseNet parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(CFG.D_CNN, d_proj),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(d_proj)

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)  # [B, 1024, H, W]
        B, C, H, W = features.shape
        
        # Spatial features: reshape to [B, H*W, C] then project
        spatial = features.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, 1024]
        spatial = self.layer_norm(self.projection(spatial))  # [B, H*W, d_proj]
        
        return spatial


print("\u2705 TextEncoder (sequence) & ImageEncoder (spatial) defined.")


```

**Tool: bash**

**Input:**
```json
{
  "command": "$nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\co_attention.ipynb\" | ConvertFrom-Json; foreach ($c in $nb.cells) { foreach ($o in $c.outputs) { if ($o.output_type -eq 'stream') { $t = ($o.text -join ''); if ($t -match 'Classification Report' -or $t -match 'TEST SET') { $t } } } }"
}
```

**Output:**
```
Evaluating on Test Set: 100%|����������| 43/43 [00:40<00:00,  1.06it/s]

============================================================
TEST SET RESULTS
============================================================
Accuracy:          0.7001
F1 (Weighted):     0.6848
F1 (Macro):        0.5714
UCE:               0.2038
============================================================

Classification Report:
              precision    recall  f1-score   support

    Negative       0.63      0.54      0.58       204
     Neutral       0.47      0.26      0.33        70
    Positive       0.75      0.86      0.80       403

    accuracy                           0.70       677
   macro avg       0.62      0.55      0.57       677
weighted avg       0.68      0.70      0.68       677


============================================================
UNCERTAINTY ANALYSIS
============================================================
Mean uncertainty (correct):   0.4731 +/- 0.1364
Mean uncertainty (incorrect): 0.5750 +/- 0.1090
============================================================


```

---

## Assistant (Plan · Kimi K3 (2x usage) · 87.0s)

_Thinking:_

This is a great comparison. The sibling `co_attention.ipynb` (single fused model, sequence-level features, proper multi-head cross attention, but still frozen backbones and no class weights) achieves:
- Accuracy 70.0%, Macro F1 0.571
- Neutral: P 0.47, R 0.26, F1 0.33 — still weak but NOT collapsed to zero.

So even with the same frozen backbones, the proper sequence-level co-attention gets neutral F1 = 0.33 vs ADEF's 0.00. This confirms the main culprits in the ADEF notebook:

1. **Degenerate "co-attention"** (pooled scalar gate) → the h_c branch is nearly useless, and the whole model relies on weak pooled unimodal heads.
2. **Class imbalance without compensation** → collapse to majority. (co_attention also suffers but less: neutral recall 0.26).
3. **Tiny LR (2e-5) for freshly-initialized heads** → undertraining (ADEF train acc only 75.9% at epoch 30 vs co_attention probably higher; anyway).
4. **tau=0.5 hardcoded and K_tv ≈ 0.014** → Route B dead.

Also to confirm: ADEF's decision path uses 3 independently-trained heads + parameter-free DS fusion; the heads trained on pooled features. The neutral class gets no evidence because features are not discriminative enough and majority class dominates the SoS loss.

Now let me design the plan. The user asked (in plan mode):

1. Fix the ADEF architecture in the notebook — explain why neutral is mispredicted.
2. Make tau (and ALL hyperparameters) part of the config.
3. Review the thesis plan md, improve it, and modify the old plan file with the best proposed model.

Since I'm in plan mode, I need to present:
- Diagnosis (why neutral fails)
- Proposed fixes for the notebook
- Proposed rewrite of the plan md
- Ask clarifying questions about tradeoffs (e.g., unfreeze backbones vs keep frozen given RTX 3060 12GB; true sequence-level co-attention is heavier — training time per epoch goes up).

Key decisions to ask the user:
1. **Encoder fine-tuning**: Keep frozen (fast, matches comparative-analysis methodology with other notebooks) vs unfreeze top layers (better features, slower, more VRAM). The other notebooks all use frozen encoders — for a fair comparative analysis, maybe ADEF should also stay frozen. This is important for thesis methodology! Let me ask.
2. **Co-attention style**: thesis-plan formula (pooled dot-product W matrix softmax) vs sequence-level multihead cross-attention like co_attention.ipynb (proven better in their own experiments). Recommend sequence-level (their own data shows it works better).
3. **Imbalance handling**: class-weighted evidential loss (recommended, minimal) vs WeightedRandomSampler vs both.
4. Whether to add fused-opinion supervision (new loss term on alpha_fusion) — changes the loss section of the thesis plan; recommend yes.

Let me also think about the "best proposed model" for the rewritten plan:

**Improved ADEF v2 design:**

1. **Encoders (sequence-level)**:
   - TextEncoder: RoBERTa-base → last_hidden_state [B, L_t, 768] → Linear+ReLU+LayerNorm → H_t [B, L_t, d]. (Optionally unfreeze last k layers.)
   - ImageEncoder: DenseNet121 → feature map [B, 1024, 7, 7] → flatten 49 patches → Linear+ReLU+LayerNorm → H_v [B, 49, d].
   - Pooled h_t (masked mean-pool), h_v (mean-pool) also produced for unimodal ENN heads.

2. **Bidirectional co-attention (true)**:
   - A = softmax_row( H_t W H_v^T / sqrt(d) ) ∈ [B, L_t, 49] and A^T (softmax over text for each patch).
   - Attended: t2v = A H_v... wait per plan: h_t·A. Fine-grained: 
     - text-attended visual summary: v_att = A^T... Let me define: A_ij = attention from token i to patch j. Row-softmax: each token distributes attention over patches → attended visual per token: V' = A H_v [B, L_t, d] → masked mean-pool → v_att. Col-softmax: each patch attends over tokens → T' = A^T... Actually simpler to follow plan's formula with proper sequence dims:
     - A = softmax(H_t W H_v^T / √d) — apply softmax on last dim (over patches) and also transpose-softmax over tokens.
     - h_c = Concat(pool(H_t A), pool(H_v A^T)) → MLP → [B, d].
   - This matches thesis formula while operating at sequence level.

3. **ENN heads** (unchanged, softplus, +1).

4. **ADEF fusion** (keep novelty, fix scale):
   - K_tv as before.
   - **tau as CFG hyperparameter** (default recalibrated: e.g., 0.3, or auto-calibrated from validation K_tv distribution — but keep simple: config with default ~0.2–0.3; make it tunable).
   - Optionally soft routing: w = sigmoid((K_tv - tau)/s) with temperature s in config — smoother, differentiable, avoids hard 0%/100% splits. I can offer this as an option (config flag `SOFT_ROUTING`). Hmm, keep simple: offer hard vs soft in config; default soft? For thesis, hard routing is easier to explain; soft is smoother. Let me propose soft with config option — ask user? Too many questions. I'll put it in the plan as config option `ROUTING = "soft" | "hard"`, default "soft", and mention rationale. Actually let me ask the user — it's a design choice that affects thesis narrative. Hmm, I can include both with config and explain in plan; no need to ask.

5. **Class imbalance fix**:
   - Compute class weights w_c = N / (K * n_c) from train set → CFG.
   - Apply in evidential SoS error term: weighted per-sample (w_y * L_err). Also optionally weighted KL.
   - Optionally WeightedRandomSampler (config flag USE_BALANCED_SAMPLER, default False if weights on).
   - Also evaluate macro-F1 as best-model selection metric instead of weighted-F1! This is important: selecting on weighted F1 favors majority-class performance and directly contributes to neutral being ignored. Change `best_val_f1` to macro F1.

6. **Fused-opinion supervision** (new): convert fused belief/uncertainty to Dirichlet alpha_f = b_f * (M/u_f) + 1 and add L(alpha_f) to L_sup. This makes the final decision directly optimized. Update plan accordingly: L_sup = Σ_k L(α_k) + L(α_fusion).

7. **LR/param groups**: heads+projections LR = CFG.LR_HEAD (1e-4 default), backbone (if unfrozen) LR = CFG.LR_BACKBONE (2e-5). Scheduler T_max from CFG.

8. **Config completeness**: move GAMMA, TAU, GRAD_CLIP, NUM_BINS_UCE, LR scheduler params, class-weight, sampler flag, routing mode, fine-tune depth (N_UNFREEZE_TEXT/N_UNFREEZE_IMAGE), evidence activation (softplus/relu), etc. into CFG.

9. **Evaluation**: keep Acc, weighted-F1, macro-F1, UCE + per-class report; add neutral-focused analysis maybe. Keep it minimal.

**Training-time considerations on RTX 3060 12GB**: sequence-level co-attention with L_t=150 and 49 patches, batch 16, frozen backbones with no_grad → memory fine (the co_attention.ipynb ran at 40s/epoch vs 56s; fine). Unfreezing will increase memory & time significantly (RoBERTa backward through last layers, DenseNet backward). With batch 16 @224px it might still fit in 12GB but slow (~2-3x). Ask user.

**Questions to ask the user:**

1. Backbone fine-tuning: (a) keep frozen like other 4 notebooks for fair comparison (recommended for methodology), (b) unfreeze last blocks for max accuracy. 
2. Co-attention implementation: (a) sequence-level multihead cross-attention (proven in co_attention.ipynb, neutral F1 0.33 vs 0.00) (recommended), (b) thesis-formula pooled dot-product style but done right. Hmm — actually the plan says A = softmax(h_t W h_v^T/√d) which IS sequence-capable; the notebook just implemented it on pooled vectors. The "faithful to thesis formula" version = sequence-level dot-product co-attention with the W matrix. The multihead version is a stronger variant. Options: (a) faithful dot-product co-attention on sequences (matches thesis formulas exactly, recommended for thesis defense), (b) multihead cross-attention (stronger but deviates from written formulas). I'd recommend (a) with option to upgrade.
3. Update the md plan file to match v2 design — confirm they want the file rewritten in the same Indonesian style (yes, "just modify the old plan" — no need to ask).

Actually also important: should the fixes keep the same dataset filtering? The filter drops text-positive+image-negative and text-negative+image-positive pairs (contradictory pairs!). Wait — this is huge for the ADEF thesis: the whole point of Route B (conflict-aware fusion) is to handle sarcasm/conflict cases, but the dataset filtering REMOVES exactly the contradictory text-image pairs (positive text + negative image and vice versa)! That's why K_tv is near zero — the training data has almost no cross-modal conflict by construction! Route B can never fire because conflicting pairs were filtered out at data loading. This is a major finding: the conflict-handling mechanism is trained/evaluated on data with conflicts removed. For the thesis on sarcasm/conflict, they should NOT filter those pairs (or at least discuss it). Hmm, but the filter might be there because those samples are noisy/ambiguous. For the comparative analysis across 5 notebooks, all notebooks probably use the same filter (need consistency). Let me check one other notebook quickly for the same filter... Actually all notebooks share the same data loading code most likely (they all were generated similarly). The filter is in the adef notebook cell 3. For the thesis's conflict story, removing contradictory pairs undermines Route B. I should mention this in the diagnosis and propose keeping them (config flag FILTER_CONFLICTS) — but that changes the dataset vs other notebooks → comparability issue. Ask the user!

Let me verify the other notebooks have the same filter quickly.

Let me verify whether the sibling notebooks share the same data filtering, and check the remaining notebooks' results for context:

**Tool: bash**

**Input:**
```json
{
  "command": "foreach ($f in @('early_fusion.ipynb','late_fusion.ipynb','cross_attention.ipynb','co_attention.ipynb')) { $nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\$f\" | ConvertFrom-Json; $hasFilter = $false; foreach ($c in $nb.cells) { $src = ($c.source -join ''); if ($src -match 'is_valid') { $hasFilter = $true } }; \"$f filter=$hasFilter\" }"
}
```

**Output:**
```
early_fusion.ipynb filter=True
late_fusion.ipynb filter=True
cross_attention.ipynb filter=True
co_attention.ipynb filter=True

```

**Tool: bash**

**Input:**
```json
{
  "command": "foreach ($f in @('early_fusion.ipynb','late_fusion.ipynb','cross_attention.ipynb')) { $nb = Get-Content -Raw \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\$f\" | ConvertFrom-Json; \"===== $f =====\"; foreach ($c in $nb.cells) { foreach ($o in $c.outputs) { if ($o.output_type -eq 'stream') { $t = ($o.text -join ''); if ($t -match 'Neutral') { ($t -split \"`n\" | Select-String -Pattern 'Accuracy:|F1 \\(Macro\\)|Neutral|Positive|Negative' | ForEach-Object { $_.Line.Trim() }) } } } } }"
}
```

**Output:**
```
===== early_fusion.ipynb =====
Accuracy:          0.6839
F1 (Macro):        0.5705
Negative       0.60      0.60      0.60       204
Neutral       0.45      0.27      0.34        70
Positive       0.75      0.80      0.77       403
===== late_fusion.ipynb =====
Accuracy:          0.6662
F1 (Macro):        0.5545
Negative       0.60      0.62      0.61       204
Neutral       0.31      0.29      0.30        70
Positive       0.76      0.75      0.76       403
===== cross_attention.ipynb =====
Accuracy:          0.6913
F1 (Macro):        0.5687
Negative       0.60      0.60      0.60       204
Neutral       0.42      0.26      0.32        70
Positive       0.76      0.81      0.79       403

```

---

## Assistant (Plan · Kimi K3 (2x usage) · 127.6s)

_Thinking:_

Now I have a complete picture:

**Benchmark comparison (all with frozen backbones, same data filter):**
| Model | Acc | Macro F1 | Neutral F1 |
|---|---|---|---|
| early_fusion | 0.684 | 0.571 | 0.34 |
| late_fusion | 0.666 | 0.555 | 0.30 |
| cross_attention | 0.691 | 0.569 | 0.32 |
| co_attention | 0.700 | 0.571 | 0.33 |
| **ADEF (thesis model)** | **0.672** | **0.432** | **0.00** ← worst! |

The thesis's proposed model is currently the WORST of all 5 models and completely fails on neutral. All models struggle with neutral (F1 0.30-0.34) due to the 10% class imbalance, but ADEF collapses entirely.

**Root causes for ADEF's neutral collapse:**

1. **Degenerate co-attention**: pooled [B,d] vectors → "attention matrix" A is a single scalar [B,1,1] with sigmoid gate. No fine-grained alignment as the thesis plan specifies. h_c branch ≈ gated copy of h_t/h_v — nearly useless.

2. **Class imbalance (59.5% pos / 30% neg / 10.4% neutral) with no compensation** — no class weights, no sampler. SoS evidential loss + argmax → majority collapse. Neutral = "absence of sentiment" class is hardest.

3. **Model selection on weighted-F1** — biased to majority; best checkpoint ignores neutral entirely.

4. **LR = 2e-5 on fresh heads** — undertrained (train acc 75.9% @ epoch 30, heads only 2.1M params). 2e-5 is a backbone-finetuning LR; frozen backbones → heads need ~1e-4.

5. **Route B dead**: K_tv mean 0.014 vs tau=0.5 hardcoded → Route A 100%. Two compounding reasons: (a) tau scale mismatch, (b) **the data filter deletes all contradictory text-image pairs** — the exact sarcasm/conflict cases ADEF was designed for. Conflict mechanism is trained on data with no conflict.

6. **Fusion output not supervised**: loss only on α_t, α_v, α_c (+L_con); p_final never appears in the loss → final decision quality depends indirectly on the three heads.

7. Neutral + EDL mismatch: p_i = b_i + u/M spreads uncertainty uniformly — for a "neutral" class, one could argue u should mass on neutral, but that's a design discussion, optional.

**Fix plan for the notebook (adef_co_attention.ipynb):**

Cell 2 (CFG): add ALL hyperparameters:
- TAU = 0.3 (config; default recalibrated to K_tv scale)
- ROUTING_MODE = "hard"/"soft", ROUTE_TEMP (softness)
- GAMMA = 1.0 (conflict loss weight)
- LR_HEADS = 1e-4, LR_BACKBONE = 2e-5, WEIGHT_DECAY
- CLASS_WEIGHTS = True/False, USE_BALANCED_SAMPLER flag
- SELECT_METRIC = "macro_f1"
- GRAD_CLIP = 1.0
- UNFREEZE_TEXT_LAYERS = 0, UNFREEZE_IMAGE_BLOCKS = 0 (config; 0 = frozen for fair comparison)
- NUM_HEADS, DROPOUT, EVIDENCE_ACT = "softplus", UCE_BINS, scheduler T_max etc.

Cell 3 (data): keep filter but add CFG flag FILTER_CONFLICT_PAIRS (document the conflict-removal issue). Maybe ask user.

Cell 5 (encoders): sequence-level encoders (like co_attention.ipynb) producing H_t [B,L,d], H_v [B,49,d] + pooled h_t, h_v for unimodal heads. Optional partial unfreezing via config.

Cell 6 (co-attention): implement TRUE bidirectional co-attention per thesis formula at sequence level:
- A = softmax(H_t W H_v^T / √d) over patches (row) and A^T softmax over tokens (col)
- v_att = masked-pool(A @ H_v... wait: per token i, attended visual = Σ_j A_ij H_v_j → [B,L,d] → masked mean pool → v_att
- t_att = per patch j, attended text = Σ_i A^T_ji H_t_i → [B,49,d] → mean pool → t_att
- h_c = MLP(Concat(t_att, v_att)) → [B,d]
This matches the thesis formulas exactly with sequences instead of pooled vectors.

Cell 8 (ADEF): tau from config; hard or soft routing via config; keep DS combination; fix u_final_b clamp with eps; route stats.

Cell 10 (loss): add class-weighted SoS term (per-sample weight w_y); add fused-opinion supervision: alpha_f = b_f*S_f + 1 with S_f = M/u_f, add L(α_f) weighted by config LAMBDA_FUSED (default 1.0). GAMMA from CFG.

Cell 12 (training): param groups (heads LR vs backbone LR), macro-F1 model selection, GAMMA/clip from CFG.

Cell 13 (eval): add per-class + routing stats already there; use CFG.TAU; maybe add neutral-recall printout. Keep minimal.

**Rewrite of rancangan_tesis_adef_co_attention.md:**
Update the plan doc to v2:
- Section II.1: sequence-level encoders (hidden states H_t [B,L_t,d], spatial H_v [B,N_v,d], N_v=49) + pooled vectors for unimodal ENN heads.
- Section II.2: co-attention formulas with sequence dims (A ∈ R^{L_t×N_v}), row/col softmax, attended summaries, h_c.
- Section II.5: tau as hyperparameter with calibration note (typical K_tv scale ≪ 0.5; recommend tuning on val or soft routing w = σ((K_tv−τ)/s)).
- Section III: add class-weighted evidential loss + fused-opinion supervision term; gamma/annealing as config.
- Add section on class imbalance handling + model selection metric (macro F1).
- Add a "hyperparameter table" with all values (config-driven).
- Note about conflict-pair filtering tradeoff (if user keeps filter, document that Route B handles residual disagreement; ideally evaluate on unfiltered too).

**Questions for the user (clarification):**

1. **Fair comparison vs max performance**: keep backbones frozen (consistent with the other 4 notebooks; recommended for comparative-analysis validity) vs unfreeze last blocks for ADEF only (better accuracy, breaks comparability, heavier on RTX 3060)?
2. **Conflict-pair filtering**: keep the is_valid filter (consistent with other notebooks) or remove it for ADEF so Route B actually sees text-image conflict (changes dataset, breaks comparability with the other 4)? Or keep filter + lower tau to observed K_tv scale?
3. **Co-attention formula**: faithful dot-product co-attention per thesis formulas on sequences (recommended, matches the written plan) vs multihead cross-attention (stronger empirically in their co_attention.ipynb)?

I'll ask these 3 questions, then present the plan. Actually per plan-mode workflow: ask clarifying questions first, then write the final plan. Let me ask the questions now since the plan depends on the answers.

Let me formulate the questions concisely.

I now have the full picture. Before writing the final plan, let me lay out the diagnosis, then ask a few key design questions.

## Diagnosis: Why ADEF collapses on Neutral (F1 = 0.00, recall = 0.00)

Benchmark across your 5 notebooks (same data, same frozen backbones):

| Model | Acc | Macro F1 | Neutral F1 |
|---|---|---|---|
| early_fusion | 0.684 | 0.571 | 0.34 |
| late_fusion | 0.666 | 0.555 | 0.30 |
| cross_attention | 0.691 | 0.569 | 0.32 |
| co_attention | 0.700 | 0.571 | 0.33 |
| **ADEF (thesis model)** | **0.672** | **0.432** | **0.00 — never predicts neutral** |

Root causes found in `adef_co_attention.ipynb`:

1. **Class imbalance ignored**: Train split is Positive 1878 (59%) / Negative 950 (30%) / Neutral 329 (10%). No class weights, no balanced sampler → sum-of-squares evidential loss collapses to the majority class.
2. **Model selection uses weighted-F1** → the "best" checkpoint is chosen by a majority-biased metric, so a checkpoint with 0% neutral recall wins.
3. **Co-attention is degenerate**: it runs on *pooled* `[B, 512]` vectors, so the "attention matrix" A is a **single scalar** `[B,1,1]` through a sigmoid — no fine-grained token↔patch alignment as your thesis plan specifies. The h_c branch is nearly useless.
4. **LR = 2e-5 on freshly-initialized heads** (backbones frozen): only 2.1M trainable params trained with a fine-tuning-scale LR → undertrained (train acc only ~76% after 30 epochs).
5. **Route B is dead code**: mean K_tv = 0.014 vs hardcoded τ = 0.5 → Route A fires 100%. Two compounding reasons: (a) τ is off-scale for observed K_tv, and (b) your `is_valid` filter **deletes exactly the contradictory text↔image pairs** (positive-text+negative-image and vice versa) — i.e., the sarcasm/conflict cases ADEF was designed for are removed from the data.
6. **Fusion output never supervised**: loss only touches α_t, α_v, α_c; p_final is not in the loss, so the final decision is only indirectly optimized.

Three design decisions affect comparability with your other 4 notebooks, so I need your call before finalizing the plan:

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "question": "Backbone fine-tuning: the other 4 notebooks all use fully-frozen RoBERTa/DenseNet. Should ADEF stay frozen (fair comparison) or unfreeze top layers (better features, breaks comparability, ~2-3x slower on your RTX 3060)?",
      "header": "Encoder fine-tuning",
      "options": [
        {
          "label": "Keep frozen (Recommended)",
          "description": "Fair comparison with the other 4 models; fix neutral via class weights + proper co-attention + higher head LR instead."
        },
        {
          "label": "Unfreeze last blocks",
          "description": "Unfreeze last 2 RoBERTa layers + last DenseNet block with small LR (2e-5). Better features, but no longer comparable to the other notebooks and heavier to train."
        },
        {
          "label": "Config-switchable, default frozen",
          "description": "Add UNFREEZE_TEXT_LAYERS / UNFREEZE_IMAGE_BLOCKS to CFG (default 0 = frozen) so you can ablate it later for the thesis."
        }
      ]
    },
    {
      "question": "Conflict-pair filtering: your is_valid filter removes all contradictory text-image pairs (the sarcasm cases ADEF's Route B exists for). This is why K_tv ≈ 0.014 and Route B never fires. What should I do?",
      "header": "Conflict data filter",
      "options": [
        {
          "label": "Keep filter + recalibrate tau (Recommended)",
          "description": "Keep dataset identical to the other 4 notebooks for fair comparison; put TAU in CFG with a realistic default (~0.2) and document the tradeoff in the plan."
        },
        {
          "label": "Remove filter for ADEF only",
          "description": "Route B sees real conflict, but dataset differs from the other 4 models (comparison becomes unfair; also adds ~330 noisy samples)."
        },
        {
          "label": "Config flag, default keep",
          "description": "Add FILTER_CONFLICT_PAIRS=True to CFG so you can ablate it in thesis experiments without editing data code."
        }
      ]
    },
    {
      "question": "Co-attention implementation for the fixed ADEF: which style should the plan and notebook use?",
      "header": "Co-attention style",
      "options": [
        {
          "label": "Thesis-formula on sequences (Recommended)",
          "description": "A = softmax(H_t W H_v^T/√d) with H_t [B,150,d], H_v [B,49,d], exactly your plan's formulas but at token/patch level instead of pooled. Faithful to the written thesis plan."
        },
        {
          "label": "Multihead cross-attention",
          "description": "nn.MultiheadAttention bidirectional like co_attention.ipynb (your best model, neutral F1 0.33). Stronger, but deviates from the written thesis formulas."
        },
        {
          "label": "Hybrid: formula + multihead option",
          "description": "Implement thesis-formula version, but expose NUM_HEADS/ATTN_TYPE in CFG so you can ablate both."
        }
      ]
    }
  ]
}
```

**Output:**
```
User has answered your questions: "Backbone fine-tuning: the other 4 notebooks all use fully-frozen RoBERTa/DenseNet. Should ADEF stay frozen (fair comparison) or unfreeze top layers (better features, breaks comparability, ~2-3x slower on your RTX 3060)?"="Keep frozen (Recommended)", "Conflict-pair filtering: your is_valid filter removes all contradictory text-image pairs (the sarcasm cases ADEF's Route B exists for). This is why K_tv ≈ 0.014 and Route B never fires. What should I do?"="Keep filter + recalibrate tau (Recommended)", "Co-attention implementation for the fixed ADEF: which style should the plan and notebook use?"="Thesis-formula on sequences (Recommended)". You can now continue with the user's answers in mind.
```

---

## Assistant (Plan · Kimi K3 (2x usage) · 144.5s)

_Thinking:_

The user chose:
1. Keep backbones frozen (fair comparison)
2. Keep conflict filter + recalibrate tau (put TAU in CFG with realistic default)
3. Thesis-formula co-attention on sequences

Now let me write the final plan. I'm still in plan mode — I present the plan and wait for approval.

Let me structure the plan:

## Plan

### Part 1: Fix `adef_co_attention.ipynb` (7 targeted cell edits)

**Cell 2 — CFG (all hyperparameters):**
Add every hyperparameter to CFG:
- `TAU = 0.2` (recalibrated to observed K_tv scale; observed mean 0.014, so even 0.2 rarely fires... hmm. Let me think about the right default for tau.

K_tv = Σ_{i≠j} b_t,i · b_v,j. Observed mean 0.014 ± 0.009, max probably ~0.05. If tau = 0.2, Route B still never fires. Hmm.

Wait — why is K_tv so low? Because both unimodal heads are trained toward the SAME label → beliefs align → conflict tiny. K_tv = (Σb_t)(Σb_v) − Σ(b_t·b_v). If b_t = b_v = b, then K_tv = (Σb)² − Σb² = 2Σ_{i<j} b_i b_j. For b = (0.5, 0.3, 0.1) (u=0.1): K = 2(0.15+0.05+0.03) = 0.46. For b=(0.8,0.05,0.05), u=0.1: K = 2(0.04+0.04+0.0025)≈0.165. Hmm but observed is 0.014?! That implies beliefs are very peaked AND aligned... e.g., b = (0.9, 0.02, 0.03): K = 2(0.018+0.027+0.0006) = 0.09. Still higher than 0.014. For K=0.014, beliefs must be nearly one-hot: b=(0.95,0.01,0.01), u≈0.03: K = 2(0.0095+0.0095+0.0001)≈0.038. Even lower needed... b=(0.97,0.005,0.005): K≈0.019. OK so beliefs are extremely peaked (near one-hot) AND modalities agree. That means heads are overconfident → aligned → no conflict. Right.

So for Route B to fire at all, tau needs to be like 0.05–0.1 on the current scale, OR the scale changes once we add class weights (less overconfident majority) and better training. With class-weighted loss, beliefs on neutral samples will be less peaked, K_tv will rise. A sensible default: TAU = 0.1 with config + calibration note: after training, pick tau as a high quantile (e.g., 80th percentile) of validation K_tv. I could add an optional `TAU = None` → auto-calibrate on val set after training? That's elegant: config `TAU_MODE = "fixed" | "auto"`, `TAU = 0.1`, `TAU_QUANTILE = 0.8`. Hmm, keep it simpler: TAU in CFG with default 0.1, plus a post-training calibration cell that reports the K_tv distribution and suggests tau. For the plan doc, specify tau as tunable hyperparameter with recommended range [0.05, 0.3] and calibration procedure.

Actually — even better and simple: keep hard routing with CFG.TAU, and in the eval cell compute K_tv quantiles to report. Default TAU = 0.1. With class weights the conflict scale will shift up anyway. Fine.

- `GAMMA = 1.0` (conflict loss weight)
- `LAMBDA_FUSED = 1.0` (weight of fused-opinion supervision loss)
- `LR_HEADS = 1e-4` (projections/co-attn/ENN heads), keep `LR = 2e-5`... better: replace LR with `LR_HEADS = 1e-4` and `LR_BACKBONE = 2e-5` (backbone unused since frozen, but keep for future). Simpler: `LR = 1e-4` since all trainable params are heads. I'll use LR = 1e-4 with comment.
- `CLASS_WEIGHTS = True`, computed from train df: w_c = N/(K·n_c) ≈ [1.11, 3.20, 0.56]
- `SELECT_METRIC = "macro"` — model selection on macro F1
- `GRAD_CLIP = 1.0`, `UCE_BINS = 10`, `NUM_HEADS`? No — user chose thesis-formula dot-product, no multihead. So no NUM_HEADS.
- `EVIDENCE_ACT = "softplus"`
- Scheduler: `SCHEDULER = "cosine"`, T_max = EPOCHS
- Keep existing: BATCH_SIZE=16, EPOCHS=30, WD, MAX_LEN, dims, DROPOUT=0.3, ANNEALING_EPOCHS=10.

**Cell 3 — data:** add `CFG.FILTER_CONFLICT_PAIRS = True` guard around is_valid (documented), compute `CLASS_WEIGHTS` tensor from train_df after split (or in cell 4). Minor.

**Cell 5 — encoders → sequence-level:**
- TextEncoder: RoBERTa (frozen, no_grad) → last_hidden_state [B,L,768] → Linear+ReLU+LayerNorm → H_t [B,L,512]; also masked mean-pool → h_t [B,512]. Return both + mask.
- ImageEncoder: DenseNet features [B,1024,7,7] → flatten 49 → Linear+ReLU+LayerNorm → H_v [B,49,512]; mean-pool → h_v [B,512].
Keep backbones frozen (user choice), keep `with torch.no_grad()`.

**Cell 6 — true bidirectional co-attention (thesis formula, sequence level):**
- W_attn: Linear(d,d,bias=False)
- S = H_t W H_v^T / √d → [B, L, 49]
- A_tv = softmax(S, dim=-1) (each token over patches); A_vt = softmax(S^T, dim=-1) (each patch over tokens) — mask padding tokens with -inf before softmax over text dim.
- t_att = masked-mean-pool(A_vt-weighted text: V2T = A_vt @ H_t → [B,49,d] → mean → [B,d])  — "visual-guided text"
- v_att = masked-mean-pool(A_tv @ H_v → [B,L,d] → masked mean pool over tokens → [B,d]) — "text-guided visual"
- h_c = LayerNorm(Linear(Concat(t_att, v_att)) + ReLU + Dropout) → [B,d]
Matches plan formulas: A = softmax(H_t W H_v^T/√d), h_c = Concat(H_t·A', H_v·A^T) pooled.

**Cell 7 — ENN heads + SL utils:** mostly unchanged; add eps-safe clamp in compute_belief_uncertainty (clamp S min). Keep.

**Cell 8 — ADEF module:** tau passed from CFG; keep hard routing (user picked "keep filter + recalibrate tau", routing style not asked... I'll keep hard routing as in the plan, with tau configurable). Fix: `u_final_b = clamp(1 - Σb, min=eps)`; also guard Route A norm clamp(1-K, min=eps). Return route mask for stats.

**Cell 9 — model:** wire new encoders (H_t, H_v, h_t, h_v) → co-attn → 3 ENN heads on (h_t, h_v, h_c) → ADEF. tau=CFG.TAU.

**Cell 10 — loss:**
- EvidentialLoss: add per-sample class weight w_y (from CFG.CLASS_WEIGHTS) multiplying the SoS error term (and optionally KL); annealing from CFG.
- New: fused-opinion supervision — convert (b_fusion, u_fusion) back to Dirichlet: S_f = K / u_f (clamp u_f ≥ eps, cap S_f for stability), α_f = b_f · S_f + 1; add LAMBDA_FUSED · L(α_f) to total. This directly optimizes the final decision including neutral.
- GAMMA from CFG.
- Keep L_con as is.

**Cell 12 — training loop:** 
- optimizer over trainable params with CFG.LR (=1e-4), cosine scheduler.
- model selection: best by **macro F1** (not weighted).
- loss = L_sup + λ_f·L(α_f) + γ·L_con.
- track macro F1 in history.

**Cell 13 — eval:** use CFG.TAU for route stats; add K_tv quantile report (median, p80, p90) to inform tau calibration; keep UCE/uncertainty analysis; add neutral recall line? classification_report covers it.

**Cell 14 — viz:** use CFG.TAU for threshold lines instead of hardcoded 0.5; add macro-F1 curve; K_tv histogram with new tau line.

### Part 2: Rewrite `rancangan_tesis_adef_co_attention.md`
Update the old plan in place (same Indonesian academic style, same structure), changing:
- §II.1: encoders now output **sequence-level** features H_t ∈ R^{L_t×d}, H_v ∈ R^{N_v×d} (N_v=49) plus pooled h_t, h_v for unimodal heads.
- §II.2: co-attention formulas with sequence dims: S = H_t W H_v^T/√d ∈ R^{L_t×N_v}, A_tv = softmax_rows(S), A_vt = softmax_cols(S); attended summaries + concat → h_c. (Fixes the mismatch where old formula implied pooled vectors.)
- §II.5: τ declared as hyperparameter (table), with calibration guidance (K_tv observed scale; tune on validation, e.g., p80 of val K_tv), keep both routes.
- §III.1: add class-weighting w_y in L_err; add fused-opinion loss term L(α_fusion) with α_fusion reconstruction formula; λ_fused, γ as hyperparameters.
- New subsection: class imbalance handling (weights formula w_c = N/(M·n_c)) + model selection by Macro-F1.
- New hyperparameter table (§IV or appendix): τ, γ, λ_fused, LR, WD, dropout, annealing epochs, batch, epochs, d_proj, annealing, activation, grad clip, etc.
- §IV: keep technical requirements; add numerical guards already used.
- Note the conflict-pair filtering tradeoff (Route B evaluated on residual disagreement; documented limitation/future work with MVSA multiple-annotator conflicts).

### Part 3: Verification
- Re-run notebook end-to-end on the RTX 3060 (batch 16, frozen backbones, sequence co-attn ≈ similar cost to co_attention.ipynb ~40-60s/epoch).
- Success criteria: neutral F1 > 0 (target ≈ 0.30-0.40, comparable to other models), macro F1 ≥ 0.55, Route B fires on a non-trivial share of samples (or tau recalibrated), UCE reported.
- Compare against the 4 baselines table.

I should present this as the final plan. Also mention expected runtime (~30 epochs × ~60-90s ≈ 30-45 min).

Let me also decide the exact default values:
- TAU = 0.1 (with calibration note). Given observed K_tv ~0.014±0.009 pre-fix; post-fix (class weights, better training) beliefs less peaked → K_tv scale rises. 0.1 is a reasonable starting point; calibration cell will tell.
- LR_HEADS = 1e-4
- CLASS_WEIGHTS = True
- LAMBDA_FUSED = 1.0
- GAMMA = 1.0
- SELECT best on macro F1.

One more consideration — the fused-opinion supervision: α_f reconstruction: S_f = K/u_f. But u_f can be near 0 in Route A after DS combination (u_tv = norm·u_t·u_v; with two-stage fusion u shrinks multiplicatively). If u_f → 0, S_f → ∞ → α_f huge → loss unstable. Guard: clamp u_f ∈ [u_min, 1] with u_min = 0.05 → S_f ≤ 60. Or clamp S_f ≤ S_max=50. I'll note eps/clamp guards in plan. Actually simpler alternative: supervise p_final with weighted cross-entropy instead of reconstructing α_f — simpler, stable, standard. But the thesis-plan-consistent approach is evidential. Hmm. The fused opinion (b,u) → expected prob p = b + u/K; a weighted CE on p_final is simple and stable. But reconstructing α_f keeps everything in EDL form (Bayes risk on fused Dirichlet). For thesis elegance: α_f = b_f·S_f+1, S_f = K/u_f with clamp. I'll propose that with clamps (u_f clamped to [0.05, 1] only for the reconstruction), default λ_f=1.0.

Alright. Also note: num_workers=0, pin_memory — fine on Windows.

One more check: masked softmax for A_vt over text tokens: need to mask padding tokens in S before softmax over dim=1 (text). Implementation detail: S_masked = S.masked_fill(text_mask==0 broadcast, -1e9). And A_tv over image patches: all 49 patches valid, no mask needed; but rows for padding tokens produce garbage v_att per token — handled by masked mean-pool over tokens. Good.

Also the pooled h_t for the text ENN head: masked mean pool of H_t (better than CLS with frozen backbone — either fine; use masked mean-pool).

Memory: H_t [16,150,512], H_v [16,49,512], S [16,150,49] — trivial. Runtime similar to co_attention.ipynb (~40s/epoch train) → ~25-30 min total. OK.

Now write the final plan message. It should be comprehensive but concise. I'll present it in a structured form and ask for approval to implement.

All three decisions locked in: **frozen backbones**, **keep filter + recalibrated τ in CFG**, **thesis-formula co-attention on sequences**. Here is the final plan.

---

# Plan: Fix ADEF Co-Attention (Neutral Collapse) + Config-Driven Hyperparameters + Rewrite Thesis Plan

## Part 1 — Root causes (to be documented in the fixed notebook)

The model never predicts neutral (P/R/F1 = 0.00) and is the worst of your 5 models on Macro-F1 (0.43 vs 0.55–0.57) because of 6 compounding issues: (1) no compensation for the 59/30/10 class imbalance, (2) best-checkpoint selection on majority-biased weighted-F1, (3) degenerate scalar-gate "co-attention" on pooled vectors, (4) LR 2e-5 undertrains the fresh heads, (5) τ=0.5 hardcoded while K_tv≈0.014 → Route B never fires, (6) fused output p_final never appears in the loss.

## Part 2 — Edits to `adef_co_attention.ipynb` (cell-by-cell)

**Cell 2 (CFG)** — move *every* hyperparameter into CFG (new entries marked +):
- `+ TAU = 0.1` (recalibrated default; observed K_tv scale ≈0.01–0.05, calibration cell will verify)
- `+ GAMMA = 1.0` (conflict-loss weight), `+ LAMBDA_FUSED = 1.0` (fused-supervision weight)
- `LR = 1e-4` (was 2e-5; heads are trained from scratch), `WEIGHT_DECAY`, `DROPOUT`, `BATCH_SIZE`, `EPOCHS`, `MAX_LEN`, `D_PROJ`, `ANNEALING_EPOCHS` (unchanged values)
- `+ USE_CLASS_WEIGHTS = True`, `+ CLASS_WEIGHTS` computed from train split → ≈ `[1.11, 3.20, 0.56]`
- `+ SELECT_METRIC = "macro"` (best checkpoint by **Macro-F1**, not weighted)
- `+ GRAD_CLIP = 1.0`, `+ UCE_BINS = 10`, `+ FILTER_CONFLICT_PAIRS = True`, `+ SCHED_TMAX = EPOCHS`

**Cell 3 (data)** — wrap `is_valid` filter behind `CFG.FILTER_CONFLICT_PAIRS` (behavior unchanged, ablatable, documented).

**Cell 5 (encoders → sequence-level)** — TextEncoder returns `H_t [B,150,512]` (projected hidden states) + masked mean-pooled `h_t [B,512]` + mask; ImageEncoder returns `H_v [B,49,512]` (7×7 DenseNet map flattened) + pooled `h_v [B,512]`. Backbones stay frozen (your choice).

**Cell 6 (true co-attention, thesis formula on sequences)** — `S = H_t·W·H_vᵀ/√d ∈ [B,150,49]`; `A_tv = softmax(S, patches)`, `A_vt = softmax(Sᵀ, tokens)` with padding-mask (−1e9) on text dim; attended summaries `v_att = masked-pool(A_tv·H_v)`, `t_att = pool(A_vt·H_t)`; `h_c = Linear+ReLU+LayerNorm+Dropout(Concat(t_att, v_att))`. Exactly the plan's `A = softmax(h_t W h_vᵀ/√d)`, `h_c = Concat(h_t·A, h_v·Aᵀ)` — now with real token/patch granularity instead of a sigmoid scalar.

**Cell 8 (ADEF)** — `tau` from CFG; numerical guards (`clamp(1−K, min=eps)`, `u_final_b` clamped to `[eps, 1]`); return route mask for stats. Routing logic itself unchanged (faithful to thesis).

**Cell 10 (loss)** —
- `EvidentialLoss` gains per-sample **class weight** `w_y` on the SoS error term (neutral samples weigh ~3.2× positive).
- **New fused-opinion supervision**: reconstruct fused Dirichlet `α_f = b_f·S_f + 1`, `S_f = K / clamp(u_f, min=0.05)` and add `LAMBDA_FUSED · L(α_f)` → the final decision is now directly optimized.
- `GAMMA`, `ANNEALING_EPOCHS` from CFG.

**Cell 12 (training)** — `loss = L_sup(α_t,α_v,α_c) + λ_f·L(α_f) + γ·L_con`; LR/WD/clip/cosine from CFG; **best model selected by Macro-F1**; history also stores macro-F1.

**Cell 13 (eval)** — route stats use `CFG.TAU`; add K_tv quantile report (median/p80/p90) to guide τ calibration for the thesis; keep Acc/F1-w/F1-macro/UCE/uncertainty analysis.

**Cell 14 (viz)** — threshold lines from `CFG.TAU` (no hardcoded 0.5); add Macro-F1 curve; keep the rest.

## Part 3 — Rewrite `rancangan_tesis_adef_co_attention.md` (in place, same Indonesian academic style)

- **§II.1**: encoders now specified as sequence-level: `H_t ∈ R^{L_t×d}`, `H_v ∈ R^{N_v×d}` (N_v=49), plus pooled `h_t, h_v` for unimodal ENN heads.
- **§II.2**: co-attention formulas corrected to sequence dims (`S ∈ R^{L_t×N_v}`, row/col softmax with text-mask, attended summaries → `h_c`).
- **§II.5**: τ declared a tunable hyperparameter with a **calibration procedure** (fix on validation K_tv quantiles; recommended search range [0.05, 0.3]); routes unchanged.
- **§III.1**: add class-weighted error term `w_y·L_err` (`w_c = N/(M·n_c)`), and new fused-opinion loss `L(α_fusion)` with `α_fusion = b_f·(M/u_f)+1`; `γ`, `λ_f` as hyperparameters.
- **New §III.4**: imbalance handling + **Macro-F1 model selection** rationale (why weighted-F1 caused neutral collapse).
- **New hyperparameter table (§IV)**: every value mirrored 1:1 with CFG (τ, γ, λ_f, LR, WD, dropout, annealing, batch, epochs, d_proj, grad-clip, activation, etc.).
- Document the conflict-pair filtering tradeoff as a stated limitation (Route B operates on residual disagreement; ablation flag provided).

## Part 4 — Verification (after your approval)

1. Re-run the full notebook end-to-end on your RTX 3060 (frozen backbones → ~40–60 s/epoch, ≈30 min total).
2. **Success criteria**: Neutral F1 > 0 (target ≈0.30–0.40, on par with the other 4 models), Macro-F1 ≥ 0.55, accuracy ≈0.68–0.72, Route B fires on a measurable share of test samples (else τ recalibrated from the quantile report), UCE reported.
3. Produce the updated 5-model comparison numbers for your thesis table.

**Files touched**: `adef_co_attention.ipynb` (cells 2,3,5,6,8,9,10,12,13,14), `rancangan_tesis_adef_co_attention.md` (full in-place rewrite). Nothing else — the other 4 notebooks stay untouched for comparability.

---

Approve and I'll implement.

---

## User

execute

---

## Assistant (Build · Kimi K3 (2x usage) · 247.7s)

_Thinking:_

The user approved the plan. Now I need to execute:

1. Edit `adef_co_attention.ipynb` cells 2, 3, 5, 6, 8, 9, 10, 12, 13, 14
2. Rewrite `rancangan_tesis_adef_co_attention.md`
3. Verify (run the notebook if possible)

Let me set up a todo list first, then start working.

Key challenge: editing a Jupyter notebook JSON. The notebook is 343KB, mostly outputs. I need to edit cell sources while preserving notebook structure. The best approach: use a Python script or PowerShell to modify the JSON programmatically. But Python isn't available as `python`... let me check for other Python launchers (py, python3). Actually earlier `python` failed. Let me check `py` launcher.

Actually, I could write the cell sources using the edit tool directly on the .ipynb JSON — but JSON escaping makes this error-prone. Better approach: use `jupyter nbconvert` or Python. Let me check what's available: `py`, `python3`, or the Python from the user's path (C:\Users\Residensi ADW\AppData\Roaming\Python\Python310 was in the tqdm warning — that's site-packages path, so Python 3.10 exists).

Let me check `py -3 --version` and `where.exe python`.

Plan of attack:
1. Check Python availability
2. Write new cell sources as separate .py text files (temp), then a Python script to patch the notebook JSON (replace sources of cells 2,3,5,6,8,9,10,12,13,14), optionally clear outputs.
3. Rewrite the md file with Write tool.
4. Run the notebook (jupyter nbconvert --execute or papermill) to verify. That takes ~30-45 min. Hmm, that's long but the user asked to execute. I have timeout limits (max ~2 hours per command but default 120s; I can set timeout up to 9007199254740991 ms). Running 30 epochs could take 30-60 minutes. That's feasible with a long timeout, but risky. Alternative: run a quick smoke test (2-3 epochs, subset of data?) to verify correctness, then let the user do the full run. But the plan said "Re-run the full notebook end-to-end... success criteria". The user said "execute". Let me try full run with a generous timeout. If it's too slow, I can reduce.

Actually, let me think about runtime: previous training was 56s/epoch train + 10s val ≈ 66s/epoch → 30 epochs ≈ 33 min. With sequence-level co-attention, slightly more compute but backbones still frozen with no_grad → similar, maybe 70-90s/epoch → ~40 min. Plus model download already cached. OK.

But wait — should I clear the old outputs when patching? Yes, since outputs will be regenerated on execution. If I execute via nbconvert with --inplace, outputs get replaced anyway.

Also need to be careful: the notebook metadata/kernelspec. nbconvert --execute --inplace will handle it.

Also first verify torch/CUDA availability in the Python env I'll use for execution.

Let me also double check one design detail: masked softmax for A_vt. S is [B, L_t, N_v]. 
- A_tv = softmax(S, dim=-1) — each text token attends over 49 patches. For padding tokens, this is garbage but we mask at pooling.
- For A_vt: softmax over text tokens (dim=1) of S^T [B, N_v, L_t]. Need to mask padding text tokens: set S[:, pad, :] = -1e9 before softmax over dim=1... Actually careful: A_vt = softmax(S.transpose(1,2), dim=-1) where the last dim is L_t. Mask: S_t = S.transpose(1,2).masked_fill(text_mask == 0 unsqueeze(1), -1e9). Then A_vt [B, N_v, L_t].
- t_att = A_vt @ H_t → [B, N_v, d] → mean over N_v → [B, d].
- v_att = A_tv @ H_v → [B, L_t, d] → masked mean over L_t → [B, d].

For the text ENN head pooled h_t: masked mean pool of H_t. For image pooled h_v: mean pool of H_v.

Now cell numbering: I'm replacing sources of cells by index. Careful that indices stay the same (I'm not adding/removing cells). Cells to replace: 2 (CFG), 3 (data), 5 (encoders), 6 (co-attn), 8 (ADEF), 9 (model), 10 (loss), 12 (training), 13 (eval), 14 (viz). Cell 7 (ENN head + SL utils) mostly unchanged but I might add small guards — keep as is to minimize changes. Actually cell 7 is fine.

Cell 4 (dataset/dataloaders) — unchanged. Cell 11 (UCE) — unchanged.

Now let me write all the new cell sources.

### Cell 2 — CFG

```python
# ============================================================
# CONFIGURATION (single source of truth for ALL hyperparameters)
# ============================================================

class CFG:

    # =========================
    # PATH
    # =========================
    ROOT_DIR = r"D:/MVSA_SINGLE"
    DATA_DIR = r"D:/MVSA_SINGLE/data"
    LABEL_PATH = r"D:/MVSA_SINGLE/labelResultAllFinal.txt"

    # =========================
    # DEVICE
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # DATA
    # =========================
    # Keep contradictory text-image pairs? False = drop them (matches the
    # other 4 baseline notebooks). Set False only for ablation studies.
    FILTER_CONFLICT_PAIRS = True

    # =========================
    # OPTIMIZATION HYPERPARAMETERS
    # =========================
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4              # heads/co-attn trained from scratch -> needs >= 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    SCHED_TMAX = EPOCHS    # CosineAnnealingLR period
    DROPOUT = 0.3
    SEED = 42

    # =========================
    # ARCHITECTURE HYPERPARAMETERS
    # =========================
    MAX_LEN = 150
    D_BERT = 768
    D_CNN = 1024
    D_PROJ = 512
    NUM_CLASSES = 3

    # =========================
    # EDL / ADEF HYPERPARAMETERS
    # =========================
    ANNEALING_EPOCHS = 10   # KL term reaches full strength at this epoch
    TAU = 0.1               # conflict threshold for ADEF routing (calibrate w/ quantile report)
    GAMMA = 1.0             # weight of semantic conflict loss L_con
    LAMBDA_FUSED = 1.0      # weight of fused-opinion supervision loss L(alpha_fusion)
    USE_CLASS_WEIGHTS = True  # class-weighted evidential loss (fixes neutral collapse)

    # =========================
    # MODEL SELECTION / EVALUATION
    # =========================
    SELECT_METRIC = "macro_f1"  # best checkpoint by MACRO F1 (not majority-biased weighted F1)
    UCE_BINS = 10

    # =========================
    # PRETRAINED MODELS
    # =========================
    TEXT_MODEL = "roberta-base"
    IMAGE_MODEL = "densenet121"

print(f"\u2705 Configuration loaded. Device: {CFG.DEVICE}")
```

### Cell 3 — data (wrap filter with flag)

Only change: 
```python
def is_valid(row):
    if row["text_label"] == "positive" and row["image_label"] == "negative":
        return False
    if row["text_label"] == "negative" and row["image_label"] == "positive":
        return False
    return True

if CFG.FILTER_CONFLICT_PAIRS:
    df = df[df.apply(is_valid, axis=1)]
    df = df.reset_index(drop=True)
```
Keep the rest identical.

### Cell 5 — encoders (sequence-level)

```python
# ============================================================
# FEATURE EXTRACTORS (Sequence-Level for Co-Attention)
# ============================================================

# RoBERTa text encoder -> token-level features + masked mean-pooled vector.
# Input:  input_ids [B, L_t], attention_mask [B, L_t]
# Output: H_t [B, L_t, d_proj] (token features),
#         h_t [B, d_proj]      (pooled for unimodal ENN head),
#         attention_mask [B, L_t]
class TextEncoder(nn.Module):

    def __init__(self, d_bert=768, d_proj=512):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(CFG.TEXT_MODEL)
        # Freeze RoBERTa parameters (fair comparison with baseline notebooks)
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(d_bert, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        H_t = self.projection(outputs.last_hidden_state)  # [B, L_t, d_proj]

        # Masked mean-pooling -> pooled text vector
        mask = attention_mask.unsqueeze(-1).float()       # [B, L_t, 1]
        h_t = (H_t * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-10)  # [B, d_proj]
        return H_t, h_t, attention_mask


# DenseNet-121 image encoder -> spatial (patch) features + pooled vector.
# Input:  image [B, 3, 224, 224]
# Output: H_v [B, N_v, d_proj] (N_v = 49 patch features),
#         h_v [B, d_proj]      (pooled for unimodal ENN head)
class ImageEncoder(nn.Module):

    def __init__(self, d_cnn=1024, d_proj=512):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = densenet.features
        # Freeze DenseNet parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(d_cnn, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.features(x)  # [B, 1024, 7, 7]
        B, C, H, W = features.shape
        H_v = features.view(B, C, H * W).permute(0, 2, 1)  # [B, N_v, 1024]
        H_v = self.projection(H_v)                          # [B, N_v, d_proj]
        h_v = H_v.mean(dim=1)                               # [B, d_proj]
        return H_v, h_v
```

### Cell 6 — BiCoAttention (thesis formula on sequences)

```python
# ============================================================
# BIDIRECTIONAL CO-ATTENTION MODULE (sequence-level, thesis formula)
# ============================================================

# Fine-grained bidirectional alignment between text tokens and image patches:
#   S    = H_t W H_v^T / sqrt(d)            -> [B, L_t, N_v]
#   A_tv = softmax(S,  dim=patches)          (each token -> over patches)
#   A_vt = softmax(S^T, dim=tokens, masked)  (each patch -> over tokens)
#   h_c  = MLP( Concat( pool(A_tv H_v), pool(A_vt H_t) ) )
#
# Input:  H_t [B, L_t, d_proj], H_v [B, N_v, d_proj], text_mask [B, L_t]
# Output: h_c [B, d_proj]
class BiCoAttention(nn.Module):

    def __init__(self, d_proj=512, dropout=0.3):
        super().__init__()
        self.d_proj = d_proj
        self.W_attn = nn.Linear(d_proj, d_proj, bias=False)
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj),
            nn.Dropout(dropout)
        )

    def forward(self, H_t, H_v, text_mask):
        # Affinity matrix: S = H_t W H_v^T / sqrt(d)
        S = torch.bmm(self.W_attn(H_t), H_v.transpose(1, 2)) / (self.d_proj ** 0.5)  # [B, L_t, N_v]

        # Text-guided visual attention: each token distributes over patches
        A_tv = F.softmax(S, dim=-1)                          # [B, L_t, N_v]
        v_att_tokens = torch.bmm(A_tv, H_v)                  # [B, L_t, d]
        mask = text_mask.unsqueeze(-1).float()               # [B, L_t, 1]
        v_att = (v_att_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-10)  # [B, d]

        # Visual-guided textual attention: each patch distributes over tokens
        pad = (text_mask == 0).unsqueeze(1)                  # [B, 1, L_t]
        S_T = S.transpose(1, 2).masked_fill(pad, -1e9)       # [B, N_v, L_t]
        A_vt = F.softmax(S_T, dim=-1)                        # [B, N_v, L_t]
        t_att = torch.bmm(A_vt, H_t).mean(dim=1)             # [B, N_v, d] -> [B, d]

        h_c = self.fusion_proj(torch.cat([t_att, v_att], dim=1))  # [B, d]
        return h_c
```

### Cell 8 — ADEF module (tau from config, guards)

Keep logic; add clamp guards; return route_a_mask too:

```python
class ADEFModule(nn.Module):

    def __init__(self, num_classes=3, tau=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def dempster_combine(self, b1, u1, b2, u2):
        eps = 1e-8
        b1_sum = torch.sum(b1, dim=1, keepdim=True)
        b2_sum = torch.sum(b2, dim=1, keepdim=True)
        C = b1_sum * b2_sum - torch.sum(b1 * b2, dim=1, keepdim=True)
        norm = 1.0 / torch.clamp(1.0 - C, min=eps)
        b_fused = norm * (b1 * b2 + b1 * u2 + b2 * u1)
        u_fused = norm * (u1 * u2)
        return b_fused, u_fused, C

    def forward(self, b_t, u_t, b_v, u_v, b_c, u_c):
        K = self.num_classes

        b_t_sum = torch.sum(b_t, dim=1, keepdim=True)
        b_v_sum = torch.sum(b_v, dim=1, keepdim=True)
        K_tv = b_t_sum * b_v_sum - torch.sum(b_t * b_v, dim=1, keepdim=True)  # [B, 1]

        route_a_mask = (K_tv <= self.tau).float()

        # ROUTE A
        b_tv_a, u_tv_a, _ = self.dempster_combine(b_t, u_t, b_v, u_v)
        b_final_a, u_final_a, _ = self.dempster_combine(b_tv_a, u_tv_a, b_c, u_c)

        # ROUTE B
        b_avg = (b_t + b_v) / 2.0
        b_final_b = (1.0 - K_tv) * b_avg + K_tv * b_c
        u_final_b = 1.0 - torch.sum(b_final_b, dim=1, keepdim=True)
        u_final_b = torch.clamp(u_final_b, min=1e-8)

        b_fusion = route_a_mask * b_final_a + (1.0 - route_a_mask) * b_final_b
        u_fusion = route_a_mask * u_final_a + (1.0 - route_a_mask) * u_final_b
        u_fusion = torch.clamp(u_fusion, min=1e-8, max=1.0)

        p_final = b_fusion + u_fusion / K

        return p_final, b_fusion, u_fusion, K_tv, route_a_mask
```

Wait, adding route_a_mask to the return changes the model's forward dict. Fine, I'm editing cell 9 and 12/13 accordingly.

### Cell 9 — model

```python
class ADEFCoAttnNet(nn.Module):

    def __init__(self, d_proj=512, num_classes=3, dropout=0.3, tau=0.1):
        super().__init__()
        self.text_encoder = TextEncoder(d_bert=CFG.D_BERT, d_proj=d_proj)
        self.image_encoder = ImageEncoder(d_cnn=CFG.D_CNN, d_proj=d_proj)
        self.co_attention = BiCoAttention(d_proj=d_proj, dropout=dropout)
        self.enn_text = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.enn_image = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.enn_coattn = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.adef = ADEFModule(num_classes=num_classes, tau=tau)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, image):
        H_t, h_t, text_mask = self.text_encoder(input_ids, attention_mask)
        H_v, h_v = self.image_encoder(image)
        h_c = self.co_attention(H_t, H_v, text_mask)

        alpha_t = self.enn_text(h_t)
        alpha_v = self.enn_image(h_v)
        alpha_c = self.enn_coattn(h_c)

        b_t, u_t, S_t = compute_belief_uncertainty(alpha_t, self.num_classes)
        b_v, u_v, S_v = compute_belief_uncertainty(alpha_v, self.num_classes)
        b_c, u_c, S_c = compute_belief_uncertainty(alpha_c, self.num_classes)

        p_final, b_fusion, u_fusion, K_tv, route_a_mask = self.adef(b_t, u_t, b_v, u_v, b_c, u_c)

        return {
            "alpha_t": alpha_t, "alpha_v": alpha_v, "alpha_c": alpha_c,
            "b_t": b_t, "u_t": u_t, "b_v": b_v, "u_v": u_v, "b_c": b_c, "u_c": u_c,
            "p_final": p_final, "b_fusion": b_fusion, "u_fusion": u_fusion,
            "K_tv": K_tv, "route_a_mask": route_a_mask
        }


model = ADEFCoAttnNet(
    d_proj=CFG.D_PROJ,
    num_classes=CFG.NUM_CLASSES,
    dropout=CFG.DROPOUT,
    tau=CFG.TAU
).to(CFG.DEVICE)
```

### Cell 10 — losses

```python
# Evidential Loss with optional per-sample class weights
class EvidentialLoss(nn.Module):

    def __init__(self, num_classes=3, annealing_epochs=10, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        # class_weights: tensor [K] or None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def bayes_risk_loss(self, alpha, y_onehot, sample_weight=None):
        alpha = torch.clamp(alpha, min=1e-10)
        S = torch.sum(alpha, dim=1, keepdim=True)
        p_hat = alpha / S
        err = torch.sum((y_onehot - p_hat) ** 2, dim=1)
        var = torch.sum(p_hat * (1.0 - p_hat) / (S + 1.0), dim=1)
        loss = err + var                                    # [B]
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    def kl_divergence_reg(self, alpha, y_onehot):
        ... same as before ...

    def forward(self, alpha, labels, epoch):
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        lambda_t = min(1.0, epoch / max(self.annealing_epochs, 1))
        sample_weight = None
        if self.class_weights is not None:
            sample_weight = self.class_weights[labels]      # [B]
        loss_err = self.bayes_risk_loss(alpha, y_onehot, sample_weight)
        loss_kl = self.kl_divergence_reg(alpha, y_onehot)
        return loss_err + lambda_t * loss_kl


# Reconstruct an equivalent Dirichlet from a fused Subjective-Logic opinion.
# b [B, K], u [B, 1]  ->  alpha [B, K] with S = K / u
def opinion_to_dirichlet(b, u, num_classes=3, u_min=0.05):
    u_c = torch.clamp(u, min=u_min, max=1.0)
    S = num_classes / u_c
    return b * S + 1.0
```

semantic_conflict_loss unchanged (but read num_classes from arg). Keep.

### Cell 12 — training

```python
class_weights = None
if CFG.USE_CLASS_WEIGHTS:
    counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    w = counts.sum() / (CFG.NUM_CLASSES * counts)
    class_weights = torch.tensor(w, dtype=torch.float32, device=CFG.DEVICE)
    print(f"Class counts: {counts.tolist()} -> weights: {np.round(w, 3).tolist()}")

criterion = EvidentialLoss(num_classes=CFG.NUM_CLASSES, annealing_epochs=CFG.ANNEALING_EPOCHS, class_weights=class_weights)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.SCHED_TMAX)
```

Training loop: add fused loss:
```python
alpha_f = opinion_to_dirichlet(out["b_fusion"], out["u_fusion"], CFG.NUM_CLASSES)
loss_f = criterion(alpha_f, labels, epoch)
loss = L_sup + CFG.LAMBDA_FUSED * loss_f + CFG.GAMMA * L_con
```
Also compute macro F1 for both train/val, select best on macro F1. History keys: add train_f1_macro, val_f1_macro (keep weighted too for comparison).

Note: loss.backward() through b_fusion/u_fusion — the fused path is differentiable except the route mask (constant per sample). Good — gradients flow into heads via DS combination. Also note alpha_f depends on K_tv mask; fine.

But wait: in Route A, u_fusion can get very small after two-stage DS (u products) → clamped at u_min=0.05 in opinion_to_dirichlet → S_f ≤ 60 → alpha_f ≤ 60-ish. lgamma(60) fine. OK.

One subtlety: with route mask hard, when route B is selected, b_final_b depends on K_tv (which depends on b_t, b_v) — gradients flow, fine.

Also track route percentage in training history? Optional; keep K_tv mean as before. I'll also add val macro.

Best model selection:
```python
val_f1_macro = f1_score(val_labels_list, val_preds, average="macro")
...
score = val_f1_macro if CFG.SELECT_METRIC == "macro_f1" else val_f1
if score > best_score:
    ...
```

### Cell 13 — eval

- use CFG.TAU for route_a
- add K_tv quantiles:
```python
qs = np.percentile(all_conflicts, [50, 80, 90, 95, 99])
print(f"K_tv quantiles p50/p80/p90/p95/p99: {qs}")
print(f"Current tau = {CFG.TAU}")
```
- keep rest; compute_uce with CFG.UCE_BINS.

### Cell 14 — viz

- replace hardcoded 0.5 with CFG.TAU in axhline/hist splits/axvline
- add macro F1 to F1 curve plot (plot val_f1_macro too)
- keep rest.

Now, implementation mechanics: I'll write a Python script that:
1. Loads the notebook JSON.
2. Replaces `cells[i]["source"]` with new source (as list of lines with \n endings, except maybe last line — Jupyter accepts string or list; I'll store as list of lines each ending with \n except last; actually simpler: store as a single string is also valid JSON for "source"? The nbformat spec allows source to be a string or list of strings. Many notebooks use list. To be safe, I'll use list-of-lines format to match typical Jupyter output. Let me check the current format: earlier I did `$c.source -join ''` which worked, suggesting source is an array. I'll write arrays of lines ending with "\n".
3. Clears all outputs and execution_counts.
4. Saves.

Then execute with jupyter nbconvert. First check `py` availability and jupyter.

For the md rewrite — full new content, in-place with Write tool (I already read it).

Let me now also draft the new markdown plan document (v2). It must keep the Indonesian academic style, update the architecture sections, add hyperparameter table, class-imbalance handling, tau calibration, macro-F1 selection, and the fused-opinion loss.

Let me write it:

---

Anda adalah **AI Lead Code Engineer** ... (keep the persona intro, adapt title: "Adaptive Evidential Fusion (ADEF) with Bidirectional Sequence-Level Co-Attention v2")

Sections:
- I. SPESIFIKASI INPUT & OUTPUT DATA (same, add note about stratified split + class distribution + conflict filter flag)
- II. STRUKTUR ARSITEKTUR MODEL (ADEFCoAttnNet v2)
  1. Unimodal Feature Extraction — sequence-level: H_t ∈ R^{L_t×d}, H_v ∈ R^{N_v×d}; pooled h_t, h_v
  2. Bidirectional Co-Attention — sequence formulas
  3. ENN Heads (same)
  4. Belief/Uncertainty (same)
  5. ADEF module (same routes; tau as hyperparameter + calibration procedure)
  6. Final decision (same)
- III. BLENDED LOSS FUNCTION
  1. Multi-Task Evidential Loss + class weights + fused-opinion loss
  2. Semantic Conflict Loss (same)
  3. Total loss: L = L_sup + λ_f L(α_f) + γ L_con
  4. (new) Class imbalance handling + model selection by Macro-F1
- IV. TABEL HYPERPARAMETER (config-driven)
- V. PERSYARATAN TEKNIS (same + numerical guards)

Let me write the md content carefully with LaTeX math like the original.

Also update the doc to explain why neutral failed and how v2 fixes it? That could be a "Changelog v2" section at the end — useful for thesis traceability. I'll add a short "CATATAN REVISI v2" section at the end listing the fixes.

Now — practical execution verification. The full run takes ~35-45 min. bash timeout max is huge (9007199254740991 ms). I'll run `jupyter nbconvert --to notebook --execute --inplace` with timeout ~3600000 ms (60 min). But first a quick smoke test: maybe run just a tiny subset? A safer approach: first run a quick syntax/shape check by executing only the definition cells + a tiny forward pass on CPU/GPU with dummy tensors. I can do that with a standalone script that imports the classes... but classes are in notebook cells. Alternative: create a small smoke-test script that extracts cells 1-10 code, execs them, builds the model with tiny dummy input, and runs one training step. That's fast (RoBERTa download cached already since notebooks ran before).

Actually simpler: run nbconvert execute on the full notebook but with EPOCHS temporarily small? No — the user wants the real run for thesis numbers. Plan: (1) smoke test via nbclient with a patched temp copy (EPOCHS=1, maybe subset)? That changes code. Hmm.

Better approach: 
1. Patch notebook.
2. Create temp copy with EPOCHS=1 to smoke-test the full pipeline quickly (~2 min: 1 epoch train 198 batches + val). Actually 1 epoch ≈ 60-70s + eval ~20s + viz. Good.
3. If smoke passes, set EPOCHS back to 30 in the real notebook and run full execution (~40 min) with long timeout.

That satisfies both correctness and the plan's verification criteria.

Let me also double-check: does the environment have jupyter? The notebooks were run before (outputs exist), so yes — likely via VS Code or jupyter. Check `py -m jupyter --version` or `jupyter --version`.

Also torch 2.5.1 + CUDA on RTX 3060 confirmed from cell 1 output.

One more consideration on determinism: cudnn.deterministic=True with multihead/bmm is fine.

Potential issue: `-1e9` masked_fill with float32 → fine. But if a row is fully masked... every patch row attends over 150 tokens; at least the CLS token is always present, so no all-masked rows. OK.

Another subtlety: In TextEncoder with frozen roberta under torch.no_grad() but model.train() mode — RoBERTa dropout active during train? dropout in frozen backbone under no_grad still applies since module in train mode → nondeterministic features. The original notebooks had the same issue; keep consistent (they all do this). With cudnn.deterministic set and seed, dropout is still RNG-driven but seeded. Not a problem to fix now — keep consistent with baselines for comparability.

Wait, actually one more important check in cell 12: the old code computed `train_f1` weighted. I'll keep weighted AND add macro. Selection on macro.

Also cell 13: `route_a = (conflict <= CFG.TAU).float()`.

Now, about patching the notebook: I'll write the new sources to a Python script file in the temp dir. The script will contain the sources as raw strings. Careful with the ✅ emoji (\u2705) — in Python source written via Write tool, I can include the emoji directly as UTF-8. The Write tool writes UTF-8. JSON dump with ensure_ascii=False keeps it readable. Fine.

Also careful: cell 14 viz has `axes[1, 3].hist(all_conflicts[all_conflicts <= 0.5]...)` etc. Replace with CFG.TAU and dynamic labels.

Let me also fix cell 13 to also print macro recall? classification_report covers it.

One more improvement for the thesis: in cell 13 also report neutral-class F1 explicitly:
```python
f1_per_class = f1_score(all_labels, all_preds, average=None)
print(f"Neutral F1: {f1_per_class[1]:.4f}")
```
Already in viz; add to report. OK.

Now write everything. Steps:
1. todowrite
2. Check python/jupyter availability (parallel with writing md? No—write md first, then script).

Actually order: 
- Task 1: rewrite md (Write tool).
- Task 2: write patch script + patch notebook.
- Task 3: smoke test (temp copy, EPOCHS=1).
- Task 4: full run.
- Task 5: report results.

Let me start. First the todo list, then check env, then write md.

For the patch script — I need to embed 10 cell sources. That's a big script (~700 lines). I'll write it to `C:\Users\ERLANG~1\AppData\Local\Temp\opencode\patch_adef.py`. The script reads the ipynb, replaces sources, clears outputs, saves. 

Cell sources in the script as triple-quoted raw strings. Careful: cell sources contain triple quotes? No. They contain backslashes (regex? no; paths use r"..." forward slashes; LaTeX? no). They contain `\u2705` escapes in strings — in raw strings `r"""..."""` the `\u2705` stays literal backslash-u which Python will interpret when the notebook cell runs? No! If I embed `"\u2705"` inside a raw triple-quoted string, it stays as literal `\u2705` in the cell source; when Jupyter executes that cell, Python parses `"\u2705"` as the ✅ char. That's exactly what the original cells contain (source shows `\u2705` literally). Good — use raw strings.

Wait, careful: original source contained `"\u2705 Imports loaded..."` as literal characters `\u2705` in the .py source, which Python interprets as escape when running. Yes, standard. So raw strings in my patch script preserve that. 

Also the LaTeX-free code has no other backslash issues.

Now, the md rewrite content. Let me draft it fully (Indonesian, matching original style):

---

```markdown
Anda adalah **AI Lead Code Engineer** yang ahli dalam bidang **Multimodal Sentiment Analysis (MSA)**, **Deep Learning**, **Evidential Deep Learning (EDL)**, dan **Teori Probabilitas Dempster-Shafer (DST)**.

Tugas Anda adalah menulis kode implementasi **PyTorch** yang bersih, optimal, terkomentari dengan baik, dan siap dijalankan (*production-ready*) untuk model Tesis S2 saya yang berjudul: **"Adaptive Evidential Fusion (ADEF) with Co-Attention"**.

Arsitektur model ini secara ketat dirancang berdasarkan rancangan resmi proposal tesis saya (Erlangga Dewa Sakti, Telkom University) di bawah bimbingan **Prof. Dr. ADIWIJAYA**. Model ini menggabungkan ekstraksi korelasi silang teks-visual menggunakan **Bidirectional Co-Attention tingkat sekuens (token–patch)** dengan kerangka kuantifikasi ketidakpastian berbasis **Evidential Deep Learning (EDL)**.

> **Revisi v2** — Dokumen ini telah direvisi berdasarkan hasil eksperimen awal: model v1 mengalami *class collapse* pada kelas Netral (F1 = 0.00) akibat (1) ketimpangan kelas tanpa kompensasi, (2) co-attention pada fitur ter-pooling (degeneratif), (3) ambang τ tidak ternormalisasi terhadap skala konflik, (4) seleksi model dengan metrik yang bias kelas mayoritas, dan (5) output fusi yang tidak tersupervisi langsung. Seluruh perbaikan dirangkum pada **Lampiran A (Catatan Revisi)**.

---

### I. SPESIFIKASI INPUT & OUTPUT DATA
(same + dataset note)
1. **Input Teks (T):** Tokenized text IDs `[batch, 150]` + attention mask.
2. **Input Gambar (V):** `[batch, 3, 224, 224]`.
3. **Output:** 3 kelas (0 negatif, 1 netral, 2 positif).
4. **Distribusi kelas (MVSA-Single setelah filtering):** ≈ 59% Positif / 30% Negatif / 10% Netral → wajib kompensasi ketimpangan kelas (lihat §III.4).
5. Pembagian data: stratified 70/15/15.

### II. STRUKTUR ARSITEKTUR MODEL (ADEFCoAttnNet v2)

#### 1. Unimodal Feature Extraction (Sequence-Level)
Text: RoBERTa-base (frozen) → hidden states H̃_t ∈ R^{L_t×768} → Linear+ReLU+LayerNorm → H_t ∈ R^{L_t×d}; pooled h_t = masked mean-pool(H_t) ∈ R^d.
Image: DenseNet121 (frozen) → feature map R^{1024×7×7} → flatten N_v=49 patches → Linear+ReLU+LayerNorm → H_v ∈ R^{N_v×d}; pooled h_v = mean(H_v).

#### 2. Bidirectional Co-Attention (Sequence-Level)
S = H_t W H_v^T / √d ∈ R^{L_t×N_v}
A_tv = softmax(S, dim=N_v), A_vt = softmax(S^T, dim=L_t) dengan masking token padding (−∞).
v_att = masked-mean-pool(A_tv H_v), t_att = mean-pool(A_vt H_t)
h_c = MLP(Concat(t_att, v_att)) ∈ R^d

#### 3-4. ENN heads + SL (same as before)

#### 5. ADEF (same routes; τ hyperparameter + kalibrasi)
- τ ∈ [0.05, 0.3], kalibrasi via kuantil K_tv validasi (mis. p80).
- Route A/B formulas unchanged.

#### 6. Final decision (same)

### III. BLENDED LOSS FUNCTION
#### 1. Multi-Task Evidential Loss dengan Pembobotan Kelas
L_sup = Σ_k L(α_k), L(α_k) = w_y · L_err(α_k) + λ_t L_KL(α_k)
w_c = N / (M · n_c)
#### 2. Supervisi Opini Fusi (baru)
α_fusion = b_fusion · (M / û) + 1, û = clamp(u_fusion, u_min, 1), u_min=0.05
L_fused = L(α_fusion)
#### 3. Semantic Conflict Loss (same)
#### 4. Total: L = L_sup + λ_f · L_fused + γ · L_con
#### 5. Penanganan Ketimpangan Kelas & Seleksi Model
- class weights, macro-F1 selection rationale.

### IV. TABEL HYPERPARAMETER (Config-Driven)
| Parameter | Simbol | Nilai Default | Keterangan |
...

### V. PERSYARATAN TEKNIS (same + clamp guards)

### Lampiran A — Catatan Revisi v2 (Root-Cause & Fix)
Table: masalah → akar penyebab → perbaikan.
```

Good. Now hyperparameter table values:
- BATCH_SIZE 16, EPOCHS 30, LR 1e-4, WEIGHT_DECAY 1e-4, GRAD_CLIP 1.0, SCHED cosine T_max=30, DROPOUT 0.3, SEED 42, MAX_LEN 150, D_PROJ 512, NUM_CLASSES 3, ANNEALING_EPOCHS 10, TAU 0.1 (kalibrasi [0.05,0.3]), GAMMA 1.0, LAMBDA_FUSED 1.0, USE_CLASS_WEIGHTS True, SELECT_METRIC macro_f1, UCE_BINS 10, u_min 0.05, FILTER_CONFLICT_PAIRS True.

Now write the actual files. First todo list.

Then:
1. Write md.
2. Write patch script.
3. Run patch script (find python first).

Let me check python: `py --version`, `py -m jupyter --version`. The site-packages path was `C:\Users\Residensi ADW\AppData\Roaming\Python\Python310\site-packages` — user-site for Python 3.10. `py` launcher should find it.

Let me go.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Rewrite rancangan_tesis_adef_co_attention.md (v2 plan, in place)",
      "status": "in_progress",
      "priority": "high"
    },
    {
      "content": "Patch adef_co_attention.ipynb (CFG all-hyperparams, seq-level encoders, true co-attention, fixed losses/training/eval/viz)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Smoke test: temp copy with EPOCHS=1, execute end-to-end",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Full run: execute notebook with EPOCHS=30, verify neutral F1/macro F1/routing",
      "status": "pending",
      "priority": "high"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Rewrite rancangan_tesis_adef_co_attention.md (v2 plan, in place)",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Patch adef_co_attention.ipynb (CFG all-hyperparams, seq-level encoders, true co-attention, fixed losses/training/eval/viz)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Smoke test: temp copy with EPOCHS=1, execute end-to-end",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Full run: execute notebook with EPOCHS=30, verify neutral F1/macro F1/routing",
    "status": "pending",
    "priority": "high"
  }
]
```

**Tool: bash**

**Input:**
```json
{
  "command": "py --version; py -m jupyter --version 2>&1 | Select-Object -First 5"
}
```

**Output:**
```
[31;1mpy: [31;1mThe term 'py' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m
[31;1mpy: [31;1mThe term 'py' is not recognized as a name of a cmdlet, function, script file, or executable program.[0m
[31;1m[31;1mCheck the spelling of the name, or if a path was included, verify that the path is correct and try again.[0m

```

---

## Assistant (Build · Kimi K3 (2x usage) · 12.1s)

_Thinking:_

No `py` either. Let me find Python. The site-packages path hints at `C:\Users\Residensi ADW\AppData\Roaming\Python\Python310\site-packages` — user site-packages, so the base install might be at `C:\Users\Residensi ADW\AppData\Local\Programs\Python\Python310\python.exe` or similar. Let me search.

**Tool: bash**

**Input:**
```json
{
  "command": "$candidates = @(\"$env:LOCALAPPDATA\\Programs\\Python\\Python310\\python.exe\", \"C:\\Python310\\python.exe\", \"C:\\Users\\Residensi ADW\\AppData\\Local\\Programs\\Python\\Python310\\python.exe\", \"C:\\Program Files\\Python310\\python.exe\"); foreach ($p in $candidates) { if (Test-Path $p) { \"FOUND: $p\" } }; Get-ChildItem \"C:\\Users\\Residensi ADW\\AppData\\Local\\Programs\\Python\" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }; Get-Command python3, jupyter -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }"
}
```

**Output:**
```
(no output)
```

---

## Assistant (Build · Kimi K3 (2x usage) · 8.6s)

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem \"C:\\Users\\Residensi ADW\" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name; Get-ChildItem \"C:\\\" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
}
```

**Output:**
```
fast_track_production
fast_track_staging
inetpub
Licenses
n8n
obe
PerfLogs
postgres
Program Files
Program Files (x86)
telu-graduation
tmp
undip_dashboard
Users
Windows

```

---

## Assistant (Build · Kimi K3 (2x usage) · 9.9s)

_Thinking:_

The current shell user seems to be different (working dir is D:\Coding\...). The notebook outputs referenced `C:\Users\Residensi ADW` but that user dir doesn't list... wait, `Get-ChildItem "C:\Users\Residensi ADW"` returned nothing (or the first command errored silently). Let me check C:\Users and find python via where/conda.

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem \"C:\\Users\" -Directory | Select-Object -ExpandProperty Name; Get-Command conda, uv, pip -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }; Get-ChildItem \"D:\\\" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
}
```

**Output:**
```
erlanggadewasakti
Public
Backup
Berkas
Coding
Dll
Game
Kuliah
Multimedia
MVSA_Single
Software

```

---

## Assistant (Build · Kimi K3 (2x usage) · 11.1s)

_Thinking:_

The user is `erlanggadewasakti` (ERLANG~1 in temp path). The notebook outputs said `C:\Users\Residensi ADW` — those outputs were generated on a different machine. Let me find Python under `C:\Users\erlanggadewasakti`.

**Tool: bash**

**Input:**
```json
{
  "command": "$dirs = @(\"C:\\Users\\erlanggadewasakti\\AppData\\Local\\Programs\\Python\", \"C:\\Users\\erlanggadewasakti\\anaconda3\", \"C:\\Users\\erlanggadewasakti\\miniconda3\", \"C:\\ProgramData\\anaconda3\", \"C:\\ProgramData\\miniconda3\"); foreach ($d in $dirs) { if (Test-Path $d) { \"EXISTS: $d\"; Get-ChildItem $d -ErrorAction SilentlyContinue | Select-Object -First 10 -ExpandProperty Name } }"
}
```

**Output:**
```
EXISTS: C:\Users\erlanggadewasakti\anaconda3
conda-meta
condabin
condarc.d
DLLs
envs
etc
include
Lib
Library
libs

```

---

## Assistant (Build · Kimi K3 (2x usage) · 10.1s)

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\python.exe\" --version; & \"C:\\Users\\erlanggadewasakti\\anaconda3\\python.exe\" -c \"import torch, transformers, torchvision, jupyter_client; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\"; Get-ChildItem \"C:\\Users\\erlanggadewasakti\\anaconda3\\envs\" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"
}
```

**Output:**
```
Python 3.13.9
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch, transformers, torchvision, jupyter_client; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch'
pytorch_gpu
.conda_envs_dir_test

```

---

## Assistant (Build · Kimi K3 (2x usage) · 17.7s)

_Thinking:_

Base anaconda has no torch, but there's a `pytorch_gpu` env. Let me check that env.

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\envs\\pytorch_gpu\\python.exe\" --version; & \"C:\\Users\\erlanggadewasakti\\anaconda3\\envs\\pytorch_gpu\\python.exe\" -c \"import torch, transformers, torchvision; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'); import transformers; print('transformers', transformers.__version__)\"; & \"C:\\Users\\erlanggadewasakti\\anaconda3\\envs\\pytorch_gpu\\python.exe\" -m jupyter --version 2>&1 | Select-Object -First 3"
}
```

**Output:**
```
Python 3.10.20
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'transformers'
[31;1mC:\Users\erlanggadewasakti\anaconda3\envs\pytorch_gpu\python.exe: No module named jupyter[0m

```

---

## Assistant (Build · Kimi K3 (2x usage) · 12.4s)

_Thinking:_

Hmm, `pytorch_gpu` env doesn't have transformers or jupyter. So the notebooks were previously run with a different env (or on a different machine — outputs referenced `C:\Users\Residensi ADW`, i.e., probably a university machine). This local machine may not be able to run the notebook as-is.

Let me check what pytorch_gpu has, and check other python installs. Maybe there's another env with torch+transformers. Let me check pip list in pytorch_gpu.

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\envs\\pytorch_gpu\\python.exe\" -m pip list 2>$null | Select-String -Pattern \"torch|transformers|jupyter|nbformat|nbconvert|nbclient|sklearn|scikit|pandas|seaborn|matplotlib|Pillow|tqdm|numpy\""
}
```

**Output:**
```

[7mnumpy[0m              2.0.1
[7mpillow[0m             11.1.0
[7mtorch[0m              2.5.1
[7mtorch[0maudio         2.5.1
[7mtorch[0mvision        0.20.1


```

---

## Assistant (Build · Kimi K3 (2x usage) · 22.5s)

_Thinking:_

The `pytorch_gpu` env has torch 2.5.1 + torchvision but no transformers, pandas, sklearn, seaborn, matplotlib, jupyter. The notebooks were clearly executed on a different machine (`C:\Users\Residensi ADW` — probably a lab machine). 

Options:
1. Install missing packages into pytorch_gpu (transformers, pandas, scikit-learn, seaborn, matplotlib, jupyter/nbclient/nbconvert). That's a big install into the user's env — per my guidelines, avoid installing outside working dir without confirmation. But the user said "execute" — which implies running the code. Installing packages into their conda env is a system change. Hmm. The guideline says "Avoid installing or deleting anything to/from outside of the current working directory. If you have to do so, ask the user for confirmation." and "If you have to install third-party tools/packages, you MUST ensure that they are installed in a virtual/isolated environment."

Options:
a. Ask the user which env they use to run these notebooks (maybe they have another env I haven't found — e.g., a venv in another location, or they run on a different machine).
b. Install into pytorch_gpu with confirmation.

Let me first search more thoroughly for other python environments on this machine (maybe a venv in the project, or another conda env, or a standalone python). Check `D:\Coding` for venvs, check `where.exe python` in a login shell context, check conda env list.

The `pytorch_gpu` env lacks transformers/pandas/jupyter. Let me search for other environments that might be the one used for these notebooks:

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\Scripts\\conda.exe\" env list 2>$null; Get-ChildItem \"D:\\Coding\" -Directory -Recurse -Depth 3 -Include \"venv\",\".venv\",\"env\" -ErrorAction SilentlyContinue | Select-Object -First 10 -ExpandProperty FullName"
}
```

**Output:**
```

# conda environments:
#
# * -> active
# + -> frozen
base                     C:\Users\erlanggadewasakti\anaconda3
pytorch_gpu              C:\Users\erlanggadewasakti\anaconda3\envs\pytorch_gpu

D:\Coding\Anaconda\Lib\venv
D:\Coding\FlyEnv-Data\env

```

---

## Assistant (Build · Kimi K3 (2x usage) · 11.3s)

_Thinking:_

Interesting: `D:\Coding\Anaconda\Lib\venv` — there might be another Anaconda installation at D:\Coding\Anaconda. Let me check for python.exe there.

**Tool: bash**

**Input:**
```json
{
  "command": "if (Test-Path \"D:\\Coding\\Anaconda\\python.exe\") { & \"D:\\Coding\\Anaconda\\python.exe\" --version; & \"D:\\Coding\\Anaconda\\python.exe\" -c \"import torch, transformers, pandas, sklearn, seaborn, matplotlib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('all imports OK')\" }; if (Test-Path \"D:\\Coding\\Anaconda\\envs\") { Get-ChildItem \"D:\\Coding\\Anaconda\\envs\" | Select-Object -ExpandProperty Name }"
}
```

**Output:**
```
Python 3.13.9
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch, transformers, pandas, sklearn, seaborn, matplotlib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('all imports OK')
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch'

```

---

## Assistant (Build · Kimi K3 (2x usage) · 16.2s)

_Thinking:_

`D:\Coding\Anaconda` is another base install without torch (it's probably a junction to the same anaconda3). No env with the full stack exists on this machine. The notebooks were run on a different machine ("Residensi ADW" user - probably a dorm/lab machine at Telkom University).

So to execute the notebook here, I'd need to install: transformers, pandas, scikit-learn, seaborn, matplotlib, jupyter (nbclient/nbconvert), ipykernel into the `pytorch_gpu` env. That's a substantial change to the user's conda env. Per guidelines, I should ask for confirmation before installing into their conda env.

Also — the dataset: `D:/MVSA_SINGLE`. The working dir shows `D:\MVSA_Single` exists! Good — data is present locally.

Let me verify the dataset path and check GPU availability with torch in pytorch_gpu env.

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\envs\\pytorch_gpu\\python.exe\" -c \"import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"; Test-Path \"D:\\MVSA_SINGLE\\labelResultAllFinal.txt\"; (Get-ChildItem \"D:\\MVSA_SINGLE\\data\" -ErrorAction SilentlyContinue | Measure-Object).Count"
}
```

**Output:**
```
cuda: True NVIDIA GeForce GTX 1650 Ti
True
9738

```

---

## Assistant (Build · Kimi K3 (2x usage) · 69.5s)

_Thinking:_

The local machine has:
- GTX 1650 Ti (4GB VRAM) — NOT the RTX 3060 12GB from the previous run. The notebooks were run on a different machine ("Residensi ADW" with RTX 3060).
- Dataset present at D:\MVSA_SINGLE.
- pytorch_gpu env: torch 2.5.1 CUDA works, but missing transformers, pandas, scikit-learn, seaborn, matplotlib, jupyter.

Executing here requires:
1. Installing ~6 packages into the user's conda env (needs confirmation per guidelines).
2. GTX 1650 Ti 4GB VRAM — the previous run was on RTX 3060 12GB. With frozen backbones + no_grad + batch 16: RoBERTa-base forward on [16,150] + DenseNet121 forward on [16,3,224,224] — memory should be modest (frozen, no gradients stored for backbones). Trainable parts are small. 4GB might be enough... RoBERTa-base forward with batch 16×150 tokens ≈ few hundred MB. DenseNet121 forward similar. Activations for trainable projection layers on sequences [16,150,512] — small. Should fit in 4GB. Training will be slower than RTX 3060 (~2-3x) → 30 epochs could take 1.5-2.5 hours.

I should ask the user: install packages into pytorch_gpu and run here (GTX 1650 Ti, slower), or just deliver the patched notebook + rewritten plan so they can run it on the other machine (RTX 3060)?

Actually, let me reconsider: the task list includes running. But the user's environment can't run it without installing packages. The right move: ask the user — quick question with options. This is a real decision point (installing into their conda env is a system-level change requiring confirmation).

Options:
1. Install missing packages into pytorch_gpu and run full training here on GTX 1650 Ti (slow: ~1.5-2.5h, 4GB VRAM risk).
2. Install + smoke test only (EPOCHS=1) here, user runs full training on the RTX 3060 machine.
3. Don't install anything — just deliver patched notebook + rewritten plan; user runs it themselves.

I'll ask. Meanwhile, I can proceed with the parts that don't need execution: rewrite the md and patch the notebook (patching only needs JSON manipulation — I can do it with base anaconda python 3.13, no special packages needed, since it's just json module).

Let me ask the question now, then proceed with md + patch while... no, I can't do work while question is pending — the question tool blocks. Actually I can ask the question first, then continue with everything after. Or do the md+patch first, then ask. Better: ask now so by the time files are ready... no, tool calls are sequential. Ask the question first — it's quick for the user.

Important finding about this machine: it has a **GTX 1650 Ti (4GB)**, not the RTX 3060 from your previous runs, and the `pytorch_gpu` conda env is missing `transformers`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, and `jupyter` (the notebooks were previously run on a different machine — "Residensi ADW"). The dataset `D:\MVSA_SINGLE` is present locally, so execution is possible — but requires installing those packages into your conda env first.

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "header": "Execution strategy",
      "options": [
        {
          "description": "Install missing packages into pytorch_gpu, run EPOCHS=1 smoke test here to verify correctness, then you run the full 30-epoch training on the RTX 3060 machine yourself.",
          "label": "Install + smoke test only (Recommended)"
        },
        {
          "description": "Install packages and run the entire 30-epoch training here on the GTX 1650 Ti 4GB (roughly 1.5-2.5h, some VRAM risk with batch 16 — may need to reduce batch size, which changes results vs baselines).",
          "label": "Install + full run here"
        },
        {
          "description": "I just deliver the patched notebook + rewritten plan; you run it on your other machine.",
          "label": "No install, deliver files only"
        }
      ],
      "question": "How should I proceed with execution on this machine?"
    }
  ]
}
```

**Output:**
```
User has answered your questions: "How should I proceed with execution on this machine?"="No install, deliver files only". You can now continue with the user's answers in mind.
```

---

## Assistant (Build · Kimi K3 (2x usage) · 17.2s)

_Thinking:_

User chose "No install, deliver files only". So I will:
1. Rewrite `rancangan_tesis_adef_co_attention.md` (v2 plan).
2. Patch `adef_co_attention.ipynb` with all the fixes.
3. Skip execution — just do a static validation of the patched notebook (JSON validity + Python syntax check of cell sources using base python's `ast` module — no torch needed for syntax check!). That's a good middle ground: verify every patched cell compiles with `compile()`/`ast.parse`.

Update todos: remove smoke test/full run, add syntax validation.

Let me now write the markdown file first.

Markdown content (Indonesian, keeping original structure and style, v2):

I'll write the complete new md. Then the patch script using base anaconda python (3.13, has json — perfect for notebook patching). Then validate JSON + compile each cell source with ast.

Let me write the md now. Keep it professional, complete, matching the original's format (headers, LaTeX math). I'll be thorough but not bloated.

Understood — files only, no installs. I'll rewrite the plan and patch the notebook, then validate everything statically (JSON + Python syntax of every cell).

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Rewrite rancangan_tesis_adef_co_attention.md (v2 plan, in place)",
      "status": "in_progress",
      "priority": "high"
    },
    {
      "content": "Patch adef_co_attention.ipynb (CFG all-hyperparams, seq-level encoders, true co-attention, fixed losses/training/eval/viz)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Static validation: notebook JSON valid + every patched cell compiles (ast)",
      "status": "pending",
      "priority": "high"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Rewrite rancangan_tesis_adef_co_attention.md (v2 plan, in place)",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Patch adef_co_attention.ipynb (CFG all-hyperparams, seq-level encoders, true co-attention, fixed losses/training/eval/viz)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Static validation: notebook JSON valid + every patched cell compiles (ast)",
    "status": "pending",
    "priority": "high"
  }
]
```

---

## Assistant (Build · Kimi K3 (2x usage) · 151.7s)

**Tool: write**

**Input:**
```json
{
  "filePath": "D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\rancangan_tesis_adef_co_attention.md",
  "content": "\nAnda adalah **AI Lead Code Engineer** yang ahli dalam bidang **Multimodal Sentiment Analysis (MSA)**, **Deep Learning**, **Evidential Deep Learning (EDL)**, dan **Teori Probabilitas Dempster-Shafer (DST)**.\n\nTugas Anda adalah menulis kode implementasi **PyTorch** yang bersih, optimal, terkomentari dengan baik, dan siap dijalankan (*production-ready*) untuk model Tesis S2 saya yang berjudul: **\"Adaptive Evidential Fusion (ADEF) with Co-Attention\"**.\n\nArsitektur model ini secara ketat dirancang berdasarkan rancangan resmi proposal tesis saya (Erlangga Dewa Sakti, Telkom University) di bawah bimbingan **Prof. Dr. ADIWIJAYA**. Model ini menggabungkan ekstraksi korelasi silang teks-visual menggunakan **Bidirectional Co-Attention tingkat sekuens (token–patch)** dengan kerangka kuantifikasi ketidakpastian berbasis **Evidential Deep Learning (EDL)**.\n\n> **Revisi v2** — Dokumen ini adalah revisi dari rancangan v1 berdasarkan hasil eksperimen awal. Model v1 mengalami *class collapse* total pada kelas **Netral** (Precision = Recall = F1 = 0.00, Macro-F1 = 0.43 — terendah di antara 5 model pembanding). Analisis akar masalah dan perbaikan yang dilakukan dirangkum pada **Lampiran A (Catatan Revisi)**. Perubahan utama v2: (1) Co-Attention dinaikkan ke tingkat sekuens (token–patch), (2) kompensasi ketimpangan kelas, (3) seluruh hyperparameter (termasuk τ) menjadi terkonfigurasi, (4) seleksi model berbasis Macro-F1, dan (5) supervisi langsung pada opini hasil fusi.\n\n---\n\n### I. SPESIFIKASI INPUT & OUTPUT DATA\n1. **Input Teks ($T$):** Tokenized text IDs dengan ukuran `[batch_size, seq_len]` dan Attention Mask `[batch_size, seq_len]`. Panjang sekuens maksimal ($L_t$) adalah **150**.\n2. **Input Gambar ($V$):** Tensor gambar RGB dengan ukuran `[batch_size, 3, 224, 224]`.\n3. **Output Prediksi ($\\hat{y}$):** Label sentimen mayoritas dari **3 kelas** (0: Negatif, 1: Netral, 2: Positif).\n4. **Distribusi Kelas (MVSA-Single setelah filtering):** sekitar **59% Positif / 30% Negatif / 10% Netral**. Ketimpangan ini **wajib** dikompensasi (lihat §III.4) — tanpa kompensasi, kelas Netral mengalami *collapse*.\n5. **Pembagian Data:** *Stratified split* 70% train / 15% validasi / 15% test dengan seed tetap, identik dengan 4 notebook baseline (early/late/cross/co-attention) demi keadilan komparasi.\n\n---\n\n### II. STRUKTUR ARSITEKTUR MODEL (ADEFCoAttnNet v2)\n\nTerapkan model PyTorch modular yang terdiri dari komponen-komponen berikut:\n\n#### 1. Unimodal Feature Extraction (Sequence-Level Feature Extraction)\nBerbeda dengan v1 yang hanya mengambil vektor ter-*pooling*, v2 mempertahankan **representasi tingkat sekuens** dari kedua encoder agar Co-Attention dapat melakukan penyelarasan halus (*fine-grained alignment*). Kedua backbone **dibekukan (*frozen*)** agar komparasi dengan 4 model baseline tetap adil.\n\n*   **Text Encoder:** Gunakan Pre-trained Language Model `RoBERTa-base`. Ambil **seluruh hidden state** dari lapisan terakhir, lalu proyeksikan setiap token dengan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi laten bersama $d = 512$:\n    $$\\mathbf{H}_t = f_{\\text{RoBERTa}}(T) \\in \\mathbb{R}^{L_t \\times d}, \\qquad L_t = 150$$\n    Vektor terpooling untuk kepala unimodal diperoleh dengan *masked mean-pooling*:\n    $$\\mathbf{h}_t = \\frac{\\sum_{l=1}^{L_t} m_l \\mathbf{H}_{t,l}}{\\sum_{l=1}^{L_t} m_l} \\in \\mathbb{R}^{d}$$\n    dengan $m_l$ adalah attention mask.\n*   **Image Encoder:** Gunakan `DenseNet121` pre-trained. Ambil **feature map spasial** sebelum Global Average Pooling, $[\\,1024 \\times 7 \\times 7\\,]$, lalu ratakan menjadi $N_v = 49$ *patch* dan proyeksikan dengan Linear Layer + Aktivasi ReLU + Layer Normalization ke dimensi $d = 512$:\n    $$\\mathbf{H}_v = f_{\\text{DenseNet}}(V) \\in \\mathbb{R}^{N_v \\times d}, \\qquad N_v = 49$$\n    Vektor terpooling untuk kepala unimodal:\n    $$\\mathbf{h}_v = \\frac{1}{N_v} \\sum_{n=1}^{N_v} \\mathbf{H}_{v,n} \\in \\mathbb{R}^{d}$$\n\n#### 2. Bidirectional Co-Attention Module (Sequence-Level)\nModul ini menangkap interaksi dan penyelarasan semantik halus dua arah antara **kata-kata teks** dan **area visual** sebelum estimasi ketidakpastian.\n\n*   **Matriks Afinitas ($S$):** Hitung skor afinitas silang menggunakan perkalian dot-product terproyeksi ($\\mathbf{W} \\in \\mathbb{R}^{d \\times d}$):\n    $$S = \\frac{\\mathbf{H}_t \\, \\mathbf{W} \\, \\mathbf{H}_v^T}{\\sqrt{d}} \\in \\mathbb{R}^{L_t \\times N_v}$$\n*   **Atensi Dua Arah:** Terapkan softmax pada kedua orientasi. Token padding teks ditutup (*masked*, $-\\infty$) sebelum softmax arah visual$\\to$teks:\n    $$A_{tv} = \\text{Softmax}_{\\text{patch}}(S) \\in \\mathbb{R}^{L_t \\times N_v}, \\qquad A_{vt} = \\text{Softmax}_{\\text{token}}(S^T) \\in \\mathbb{R}^{N_v \\times L_t}$$\n*   **Representasi Teratensi:** Setiap kata menghadirkan ringkasan visual teratensi (*text-guided visual*); setiap patch menghadirkan ringkasan tekstual teratensi (*visual-guided text*):\n    $$\\mathbf{v}_{\\text{att}} = \\text{MaskedMeanPool}(A_{tv} \\, \\mathbf{H}_v) \\in \\mathbb{R}^{d}, \\qquad \\mathbf{t}_{\\text{att}} = \\text{MeanPool}(A_{vt} \\, \\mathbf{H}_t) \\in \\mathbb{R}^{d}$$\n*   **Representasi Gabungan ($\\mathbf{h}_c$):** Gabungkan kedua ringkasan lalu proyeksikan (Linear + ReLU + LayerNorm + Dropout):\n    $$\\mathbf{h}_c = \\text{MLP}\\big(\\text{Concat}(\\mathbf{t}_{\\text{att}}, \\mathbf{v}_{\\text{att}})\\big) \\in \\mathbb{R}^{d}$$\n    $\\mathbf{h}_c$ bertindak sebagai jalur ketiga yang merepresentasikan hubungan interaksi teks-gambar.\n\n#### 3. Evidential Neural Network (ENN) Heads\nAlih-alih aktivasi Softmax deterministik yang menyebabkan *overconfidence*, ketiga fitur terpooling ($\\mathbf{h}_t$, $\\mathbf{h}_v$, $\\mathbf{h}_c$) dilewatkan secara paralel ke tiga ENN Head independen (Text ENN, Image ENN, Co-Attention ENN).\n*   **Ekstraksi Bukti (*Evidence Extraction*):** Lapisan fully-connected ($d \\to d/2 \\to M$) dengan aktivasi non-negatif **Softplus** untuk menjamin $e \\ge 0$. Untuk setiap jalur $k \\in \\{t, v, c\\}$ dan kelas $i \\in \\{1, 2, 3\\}$:\n    $$e_{k,i} = \\text{Softplus}(\\mathbf{W}_k \\mathbf{h}_k + \\mathbf{b}_k)$$\n*   **Parameter Distribusi Dirichlet ($\\alpha$):**\n    $$\\alpha_{k,i} = e_{k,i} + 1$$\n*   **Kekuatan Bukti Total ($S_k$):**\n    $$S_k = \\sum_{i=1}^M \\alpha_{k,i}, \\qquad M = 3$$\n\n#### 4. Kuantifikasi Belief Mass & Uncertainty Mass\nBerdasarkan *Subjective Logic* (SL), hitung massa keyakinan dan massa ketidakpastian global secara independen untuk masing-masing jalur:\n*   **Belief Mass ($b_{k,i}$):**\n    $$b_{k,i} = \\frac{e_{k,i}}{S_k}$$\n*   **Uncertainty Mass ($u_k$):**\n    $$u_k = \\frac{M}{S_k}$$\n*   Aksioma Subjective Logic:\n    $$\\sum_{i=1}^M b_{k,i} + u_k = 1$$\n\n#### 5. Modul Adaptive Evidential Fusion (ADEF)\nModul ini melakukan fusi adaptif dengan mengevaluasi tingkat perselisihan opini antara jalur teks murni dan gambar murni.\n\n*   **Kalkulasi Massa Konflik ($K_{tv}$):**\n    $$K_{tv} = \\sum_{i=1}^M \\sum_{\\substack{j=1 \\\\ j \\neq i}}^M b_{t,i} \\cdot b_{v,j}$$\n*   **Ambang Batas $\\tau$ sebagai Hyperparameter Terkalibrasi:** $\\tau$ **bukan** konstanta yang di-hardcode. Nilainya dikalibrasi terhadap skala empiris $K_{tv}$ pada data validasi (lihat §IV). Rentang pencarian yang direkomendasikan: $\\tau \\in [0.05,\\, 0.3]$; default awal $\\tau = 0.1$. (Catatan v1: $\\tau = 0.5$ tidak pernah terlampaui karena rata-rata $K_{tv} \\approx 0.014$, sehingga Rute B tidak pernah aktif.)\n\n*   **Rute A: Fusi Normal (Dempster-Shafer Standard) — Jika $K_{tv} \\le \\tau$:**\n    *   *Tahap 1 (Fusi Teks-Visual):*\n        $$b_{tv,i} = \\frac{1}{1 - K_{tv}} (b_{t,i} \\cdot b_{v,i} + b_{t,i} \\cdot u_v + b_{v,i} \\cdot u_t), \\qquad u_{tv} = \\frac{1}{1 - K_{tv}} (u_t \\cdot u_v)$$\n    *   *Tahap 2 (Konsensus Akhir dengan Co-Attention):* Hitung konflik baru $K_{tvc}$ antara opini $(b_{tv}, u_{tv})$ dan jalur Co-Attention $(b_c, u_c)$:\n        $$K_{tvc} = \\sum_{i=1}^M \\sum_{\\substack{j=1 \\\\ j \\neq i}}^M b_{tv,i} \\cdot b_{c,j}$$\n        $$b_{\\text{fusion},i} = \\frac{1}{1 - K_{tvc}} (b_{tv,i} \\cdot b_{c,i} + b_{tv,i} \\cdot u_c + b_{c,i} \\cdot u_{tv}), \\qquad u_{\\text{fusion}} = \\frac{1}{1 - K_{tvc}} (u_{tv} \\cdot u_c)$$\n\n*   **Rute B: Fusi Resolusi Konflik (Conflict-Aware Fusion) — Jika $K_{tv} > \\tau$:**\n    Aturan ortogonal Dempster standar berisiko gagal pada kontradiksi ekstrem (*Zadeh's Paradox*). Gunakan $K_{tv}$ sebagai tuas pengontrol dinamis untuk meredam opini unimodal yang bertentangan dan mengalihkan dominasi bobot kepada representasi relasional dari **Co-Attention**:\n    $$b_{\\text{fusion},i} = (1 - K_{tv}) \\cdot \\left( \\frac{b_{t,i} + b_{v,i}}{2} \\right) + K_{tv} \\cdot b_{c,i}$$\n    $$u_{\\text{fusion}} = \\max\\Big(\\epsilon,\\; 1 - \\sum_{i=1}^M b_{\\text{fusion},i}\\Big)$$\n\n#### 6. Final Decision Making\n*   **Final Expectation Probability ($p_i$):**\n    $$p_i = b_{\\text{fusion},i} + \\frac{u_{\\text{fusion}}}{M}$$\n*   **Output Prediksi Sentimen Final ($\\hat{y}$):**\n    $$\\hat{y} = \\text{Argmax}(p_1, p_2, p_3)$$\n\n---\n\n### III. BLENDED LOSS FUNCTION (TRAINING OBJECTIVE)\n\nModel dilatih dengan *Multi-Task Learning* yang meminimalkan gabungan kerugian evidensial tersupervisi, supervisi opini fusi, dan penalti konflik:\n\n#### 1. Multi-Task Evidential Loss dengan Pembobotan Kelas ($L_{\\text{sup}}$)\n$$L_{\\text{sup}} = L(\\alpha_t) + L(\\alpha_v) + L(\\alpha_c)$$\nUntuk setiap jalur $k$:\n$$L(\\alpha_k) = w_y \\cdot L_{\\text{err}}(\\alpha_k) + \\lambda_t L_{\\text{KL}}(\\alpha_k)$$\n*   **Pembobotan Kelas ($w_y$) — BARU di v2:** Sampel ditimbang berbanding terbalik dengan frekuensi kelasnya agar kelas minoritas (Netral, $\\approx$10% data) tidak tenggelam oleh mayoritas:\n    $$w_c = \\frac{N}{M \\cdot n_c}, \\qquad w_y = w_{c=y}$$\n    ($N$ = jumlah sampel train, $n_c$ = jumlah sampel kelas $c$; menghasilkan $w \\approx [1.11,\\ 3.20,\\ 0.56]$ untuk Negatif/Netral/Positif.)\n*   **Error Term ($L_{\\text{err}}$):**\n    $$L_{\\text{err}}(\\alpha_{k}) = \\sum_{j=1}^M (y_j - \\hat{p}_{k,j})^2 + \\frac{\\hat{p}_{k,j}(1 - \\hat{p}_{k,j})}{S_k + 1}, \\qquad \\hat{p}_{k,j} = \\frac{\\alpha_{k,j}}{S_k}$$\n*   **Regularisasi KL-Divergence ($L_{\\text{KL}}$):**\n    $$L_{\\text{KL}}(\\alpha_k) = \\log \\left( \\frac{\\Gamma(\\sum_{j=1}^M \\tilde{\\alpha}_{k,j})}{\\Gamma(M) \\prod_{j=1}^M \\Gamma(\\tilde{\\alpha}_{k,j})} \\right) + \\sum_{j=1}^M (\\tilde{\\alpha}_{k,j} - 1) \\left[ \\psi(\\tilde{\\alpha}_{k,j}) - \\psi\\left(\\sum_{l=1}^M \\tilde{\\alpha}_{k,l}\\right) \\right]$$\n    dengan $\\tilde{\\alpha}_{k} = y + (1 - y) \\odot \\alpha_k$.\n*   **Annealing Coefficient ($\\lambda_t$):**\n    $$\\lambda_t = \\min\\Big(1.0,\\ \\frac{t}{\\text{ANNEALING\\_EPOCHS}}\\Big), \\qquad \\text{ANNEALING\\_EPOCHS} = 10$$\n\n#### 2. Supervisi Opini Fusi ($L_{\\text{fused}}$) — BARU di v2\nPada v1, output fusi final ($p_{\\text{fusion}}$) tidak pernah muncul di dalam loss, sehingga kualitas keputusan akhir hanya dioptimalkan secara tidak langsung. Di v2, opini hasil fusi **dikonversi kembali** ke distribusi Dirichlet ekuivalen dan disupervisi langsung dengan Evidential Loss yang sama:\n$$S_{\\text{fusion}} = \\frac{M}{\\hat{u}}, \\qquad \\hat{u} = \\text{clamp}(u_{\\text{fusion}},\\ u_{\\min},\\ 1), \\qquad \\alpha_{\\text{fusion},i} = b_{\\text{fusion},i} \\cdot S_{\\text{fusion}} + 1$$\n$$L_{\\text{fused}} = L(\\alpha_{\\text{fusion}})$$\ndengan $u_{\\min} = 0.05$ untuk stabilitas numerik. Gradien mengalir melalui aturan kombinasi Dempster ke seluruh ENN head.\n\n#### 3. Semantic Conflict Loss ($L_{\\text{con}}$)\nUntuk memandu encoder mendeteksi inkongruensi emosi dan memetakan konflik semantik menjadi ketidakpastian multimodal:\n$$L_{\\text{con}} = d_{\\text{PD}} \\cdot d_{\\text{CC}} = \\frac{1}{2} (1 - u_t) (1 - u_v) \\sum_{i=1}^M |p_{t,i} - p_{v,i}|$$\n\n#### 4. Total Loss Keseluruhan Jaringan\n$$L_{\\text{overall}} = L_{\\text{sup}} + \\lambda_f \\, L_{\\text{fused}} + \\gamma \\, L_{\\text{con}}$$\ndengan $\\lambda_f$ (default: 1.0) dan $\\gamma$ (default: 1.0) sebagai hyperparameter balancing yang terkonfigurasi.\n\n#### 5. Penanganan Ketimpangan Kelas & Seleksi Model — BARU di v2\n1.  **Pembobotan kelas** pada $L_{\\text{err}}$ (§III.1) — kompensasi utama ketimpangan 59/30/10.\n2.  **Seleksi checkpoint terbaik berdasarkan Macro-F1 validasi**, *bukan* Weighted-F1. Weighted-F1 didominasi kelas mayoritas sehingga checkpoint dengan recall Netral = 0% justru terpilih sebagai \"terbaik\" pada v1. Macro-F1 memberi bobot setara ke setiap kelas.\n3.  **Pelaporan metrik** wajib menyertakan Accuracy, Weighted-F1, **Macro-F1**, dan **F1 per kelas** agar kegagalan pada satu kelas tidak tersembunyi.\n\n---\n\n### IV. TABEL HYPERPARAMETER (SEMUA TERKONFIGURASI VIA `CFG`)\n\nSeluruh hyperparameter — termasuk $\\tau$ — dideklarasikan di satu kelas konfigurasi (`CFG`) sebagai *single source of truth*. Tidak boleh ada angka *hardcoded* di dalam definisi model, loss, training loop, maupun evaluasi.\n\n| Kategori | Parameter (`CFG`) | Simbol | Nilai Default | Keterangan |\n|---|---|---|---|---|\n| Data | `MAX_LEN` | $L_t$ | 150 | Panjang maksimal sekuens teks |\n| Data | `FILTER_CONFLICT_PAIRS` | — | `True` | Membuang pasangan teks-gambar yang kontradiktif (konsisten dgn 4 baseline; set `False` untuk ablasi) |\n| Arsitektur | `D_BERT` / `D_CNN` | — | 768 / 1024 | Dimensi output RoBERTa-base / DenseNet121 |\n| Arsitektur | `D_PROJ` | $d$ | 512 | Dimensi laten bersama |\n| Arsitektur | `NUM_CLASSES` | $M$ | 3 | Jumlah kelas sentimen |\n| Arsitektur | `DROPOUT` | — | 0.3 | Dropout pada proyeksi & ENN head |\n| Optimisasi | `BATCH_SIZE` | — | 16 | Ukuran batch |\n| Optimisasi | `EPOCHS` | — | 30 | Jumlah epoch |\n| Optimisasi | `LR` | — | $1 \\times 10^{-4}$ | Learning rate AdamW untuk seluruh parameter terlatih (backbone beku). **Catatan v1:** $2 \\times 10^{-5}$ terlalu kecil untuk head yang diinisialisasi acak → *undertraining* |\n| Optimisasi | `WEIGHT_DECAY` | — | $1 \\times 10^{-4}$ | Weight decay AdamW |\n| Optimisasi | `GRAD_CLIP` | — | 1.0 | Gradient clipping (max norm) |\n| Optimisasi | `SCHED_TMAX` | — | 30 | Periode CosineAnnealingLR |\n| Optimisasi | `SEED` | — | 42 | Seed reprodusibilitas |\n| EDL | `ANNEALING_EPOCHS` | — | 10 | Epoch saat $\\lambda_t$ mencapai 1.0 |\n| EDL | `USE_CLASS_WEIGHTS` | — | `True` | Aktifkan pembobotan kelas $w_y$ pada $L_{\\text{err}}$ |\n| EDL | `U_MIN` | $u_{\\min}$ | 0.05 | Batas bawah $u_{\\text{fusion}}$ untuk rekonstruksi $\\alpha_{\\text{fusion}}$ |\n| ADEF | **`TAU`** | $\\tau$ | **0.1** | Ambang konflik untuk routing dinamis. Kalibrasi via kuantil $K_{tv}$ validasi (mis. persentil-80); rentang pencarian $[0.05, 0.3]$ |\n| Loss | `LAMBDA_FUSED` | $\\lambda_f$ | 1.0 | Bobot supervisi opini fusi $L_{\\text{fused}}$ |\n| Loss | `GAMMA` | $\\gamma$ | 1.0 | Bobot Semantic Conflict Loss $L_{\\text{con}}$ |\n| Evaluasi | `SELECT_METRIC` | — | `\"macro_f1\"` | Metrik seleksi checkpoint terbaik |\n| Evaluasi | `UCE_BINS` | — | 10 | Jumlah bin untuk kalibrasi UCE |\n\n**Prosedur Kalibrasi $\\tau$:** (1) Latih model dengan $\\tau = 0.1$. (2) Hitung distribusi $K_{tv}$ pada himpunan validasi/test; laporkan kuantil p50/p80/p90/p95/p99. (3) Jika Rute B aktif pada $<1\\%$ atau $>50\\%$ sampel, geser $\\tau$ menuju persentil-80 dan ulangi evaluasi. (4) Laporkan sensitivitas performa terhadap $\\tau$ sebagai analisis ablasi tesis.\n\n---\n\n### V. PERSYARATAN TEKNIS IMPLEMENTASI KODE\n1.  **Stabilitas Numerik:** Epsilon $10^{-8}$ pada seluruh pembagian; `torch.clamp` untuk input log/gamma/digamma; normalisasi Dempster menggunakan `1 / clamp(1 - K, min=ε)`; $u_{\\text{fusion}}$ di-clamp ke $[\\epsilon, 1]$; masking token padding dengan $-10^9$ sebelum softmax arah visual→teks.\n2.  **Modularitas:** Kelas PyTorch rapi (`nn.Module`): `TextEncoder`, `ImageEncoder`, `BiCoAttention`, `ENNHead`, `ADEFModule`, `EvidentialLoss`. Fungsi utilitas terpisah untuk Subjective Logic (`compute_belief_uncertainty`), konversi opini→Dirichlet (`opinion_to_dirichlet`), dan Semantic Conflict Loss.\n3.  **Backbone Beku:** RoBERTa dan DenseNet121 dibekukan (`requires_grad=False` + `torch.no_grad()`) agar adil terhadap 4 baseline dan hemat VRAM.\n4.  **Metrik Evaluasi:** Accuracy, Weighted-F1, **Macro-F1**, **F1 per kelas**, classification report lengkap, confusion matrix, **Expected Uncertainty Calibration Error (UCE)**, statistik routing ADEF (% Rute A/B, rata-rata $K_{tv}$, kuantil $K_{tv}$), serta analisis ketidakpastian (mean $u$ prediksi benar vs salah).\n5.  **Device-Aware:** `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.\n6.  **Reprodusibilitas:** Seed tetap (42) untuk `random`, `numpy`, `torch`, dan cuDNN deterministik.\n\nTuliskan implementasi kode ini dengan lengkap, profesional, terstruktur, dan bersih!\n\n---\n\n### Lampiran A — Catatan Revisi v2: Analisis Akar Masalah & Perbaikan\n\nHasil eksperimen v1 pada test set: **Accuracy 0.672, Macro-F1 0.432, dan kelas Netral F1 = 0.00 (tidak pernah diprediksi)** — terendah di antara 5 model (baseline lain mencapai Macro-F1 0.55–0.57 dan F1 Netral 0.30–0.34). Rute B ADEF aktif pada **0%** sampel.\n\n| # | Gejala | Akar Masalah (v1) | Perbaikan (v2) |\n|---|---|---|---|\n| 1 | Kelas Netral tidak pernah diprediksi | Ketimpangan kelas 59/30/10 tanpa kompensasi; loss Sum-of-Squares didominasi kelas mayoritas | Pembobotan kelas $w_y$ pada $L_{\\text{err}}$ (§III.1); `USE_CLASS_WEIGHTS` |\n| 2 | Checkpoint \"terbaik\" mengabaikan Netral | Seleksi model memakai Weighted-F1 yang bias ke kelas mayoritas | Seleksi berbasis **Macro-F1** (`SELECT_METRIC`) + laporan F1 per kelas |\n| 3 | Jalur Co-Attention lemah | \"Co-Attention\" v1 beroperasi pada vektor ter-*pooling* $[B, d]$: matriks atensi $A$ berdegenerasi menjadi **skalar** $[B,1,1]$ dengan sigmoid — bukan penyelarasan token–patch | Co-Attention tingkat sekuens sesuai rumus asli: $S = \\mathbf{H}_t \\mathbf{W} \\mathbf{H}_v^T / \\sqrt{d} \\in \\mathbb{R}^{L_t \\times N_v}$ dengan softmax dua arah + masking (§II.2) |\n| 4 | Rute B tidak pernah aktif (0% sampel) | $\\tau = 0.5$ hardcoded, sedangkan rata-rata $K_{tv} \\approx 0.014$; data konflik juga terfilter | $\\tau$ menjadi hyperparameter terkonfigurasi (`CFG.TAU = 0.1`) + prosedur kalibrasi kuantil (§IV); `FILTER_CONFLICT_PAIRS` sebagai flag ablasi |\n| 5 | Keputusan fusi tidak teroptimasi | Loss hanya menyentuh $\\alpha_t, \\alpha_v, \\alpha_c$; $p_{\\text{fusion}}$ tidak ada di dalam loss | Supervisi opini fusi $L_{\\text{fused}}$ via rekonstruksi $\\alpha_{\\text{fusion}}$ (§III.2) |\n| 6 | Konvergensi lambat (train acc 76% @ epoch 30) | LR $2 \\times 10^{-5}$ untuk head acak (LR skala fine-tuning backbone, padahal backbone beku) | `LR = 1 \\times 10^{-4}` untuk seluruh parameter terlatih (§IV) |\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · Kimi K3 (2x usage) · 378.6s)

_Thinking:_

The md plan is rewritten. Now patch the notebook. I'll write a Python patch script (using base anaconda Python 3.13 which only needs the `json` module) that:
1. Loads the notebook
2. Replaces sources of cells 2, 3, 5, 6, 8, 9, 10, 12, 13, 14
3. Clears all outputs + execution_count
4. Saves

I'll write the patch script to the temp dir. The script contains the new cell sources as raw triple-quoted strings. Careful points:
- Cell sources contain `\u2705` escape sequences (literal backslash-u in source). Use raw strings r'''...''' so they stay literal — Python interprets them when the notebook cell runs. ✅
- Cell sources contain triple double-quotes? No. Contain `'''`? No.
- Source as list of lines (nbformat standard). I'll split with `splitlines(keepends=True)`.

Let me now write out all 10 cell sources completely and carefully.

Cell 2 (CFG): as drafted, plus U_MIN.

Cell 3 (data): original with filter wrapped in flag. I need to reproduce the whole cell with the small change. Let me reconstruct it exactly from what I read, with the modification:

```python
import pandas as pd
import os
# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CFG.LABEL_PATH, header=0, sep=",")
df.columns = ["id", "text_label", "image_label", "final_label"]

# Filter pasangan kontradiktif (text+ vs image- dan sebaliknya).
# Default True agar identik dengan 4 notebook baseline (fair comparison).
# Set CFG.FILTER_CONFLICT_PAIRS = False untuk ablasi konflik (Rute B ADEF).
def is_valid(row):
    if row["text_label"] == "positive" and row["image_label"] == "negative":
        return False
    if row["text_label"] == "negative" and row["image_label"] == "positive":
        return False
    return True

if CFG.FILTER_CONFLICT_PAIRS:
    df = df[df.apply(is_valid, axis=1)]
    df = df.reset_index(drop=True)

print(f"Dataset size after filtering: {len(df)}")
... (rest identical)
```

Cell 5 (encoders): new sequence-level version (drafted above).

Cell 6 (BiCoAttention): new (drafted above).

Cell 8 (ADEFModule): with tau, guards, returns route_a_mask.

Cell 9 (model): wired for sequence encoders; tau=CFG.TAU; prints params.

Cell 10 (losses): EvidentialLoss with class_weights buffer; opinion_to_dirichlet; semantic_conflict_loss unchanged.

Cell 12 (training): class weights computation, criterion, optimizer CFG.LR, scheduler CFG.SCHED_TMAX, fused loss, macro-F1 tracking + selection, GRAD_CLIP from CFG, GAMMA/LAMBDA_FUSED from CFG.

Cell 13 (eval): CFG.TAU for routes, K_tv quantiles, per-class F1 print, UCE with CFG.UCE_BINS.

Cell 14 (viz): CFG.TAU in plots, add macro-F1 curve.

Let me write these carefully now, complete and consistent.

One check for cell 12: history dict — I'll keep original keys and add "train_f1_macro", "val_f1_macro". In print, show both weighted and macro. Selection: best_score = val macro f1.

Also I realize for `best_model_state` message: "New best model! (Val Macro-F1: ...)".

Cell 13 details:

```python
# ============================================================
# TEST SET EVALUATION
# ============================================================

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("\u2705 Best model loaded for evaluation.")

model.eval()
all_preds, all_labels, all_uncertainties, all_conflicts = [], [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        ...
        route stats via out["route_a_mask"]? Easier: collect out["route_a_mask"].squeeze(1)
```

Use out["route_a_mask"] directly instead of recomputing with threshold — consistent.

Then metrics + routing stats + K_tv quantiles + per-class F1 emphasis + uncertainty analysis.

Cell 14 viz: 
- Plot 3 (F1): plot train/val macro F1 as main curves, and weighted dashed.
- Plot 4 (conflict): axhline at CFG.TAU.
- Plot 8 (routing distribution): split histograms by CFG.TAU, axvline CFG.TAU, label f"tau={CFG.TAU}".

Now write the patch script. I'll generate it as a single Python file with a dict {cell_index: source}. Use raw strings. One caution: cell sources contain backslashes only in `\u2705` escapes and `r"D:/MVSA_SINGLE"` (forward slashes, fine). In raw triple-quoted strings, `\u2705` remains literal — good.

Also must be careful that no cell source contains the sequence `'''`. Check: no triple quotes in my sources. OK.

The script:

```python
import json, io

NB = r"D:\Coding\Project\All Thesis\Comparative Analysis of Cross-Modal Fusion\adef_co_attention.ipynb"

cells_src = {}

cells_src[2] = r'''...'''
...

with io.open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, src in cells_src.items():
    assert nb['cells'][idx]['cell_type'] == 'code', idx
    nb['cells'][idx]['source'] = src.splitlines(keepends=True)
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

with io.open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('patched OK')
```

Wait — `splitlines(keepends=True)` on a string starting with a newline? My raw strings will start immediately with content. One subtlety: the last line won't have trailing newline — standard. Also nbformat accepts source as list of strings. Good.

Also the original notebook's json.dump format: originally probably written with indent 1 (jupyter standard). Using indent=1 keeps the file readable. ensure_ascii=False keeps ✅ readable (the original had literal \u2705 in code strings, which json.dump with ensure_ascii=True would escape as \\u2705... hmm wait).

Careful: the cell source contains the 7 characters `\u2705` (backslash u 2 7 0 5) inside a Python string literal. In JSON, that backslash must be escaped as `\\u2705`. json.dump handles that automatically. When the notebook loads, cell source again has `\u2705` literal, and when the cell runs, Python interprets it as ✅. Correct behavior preserved.

If I use ensure_ascii=False, the literal backslash sequences still get escaped as `\\u2705` in JSON (backslash escaping is required regardless), while actual emoji characters (like ✅ if any existed literally) would stay raw. My sources only have the escape sequences, no literal emoji. Either way works. I'll use ensure_ascii=False, indent=1.

Hmm, one more check — cell 14 in the original references `all_route_a_pct` — I changed to not need it in cell 13? Original cell 13 collected all_route_a_pct and used in routing statistics. I'll keep collecting route mask from out["route_a_mask"] and use it. Cell 14 uses only all_conflicts + CFG.TAU for the histogram. Good.

Also cell 13 originally printed route_a_total using all_route_a_pct. I'll compute from the collected route mask array.

Now write the full patch script. Let me carefully compose each cell source.

Also, keep in cell 9 the parameter counting print block.

Cell 10 — semantic_conflict_loss keep as original (with num_classes param). The fused loss criterion reuse: criterion(alpha_f, labels, epoch) — class weights also apply to fused loss. Good (weighted supervision on final decision helps neutral too).

Cell 12 — full training loop. Let me write it completely:

```python
# ============================================================
# TRAINING & VALIDATION (ADEF Blended Loss v2)
# L_overall = L_sup + LAMBDA_FUSED * L_fused + GAMMA * L_con
# ============================================================

# ---- Class weights (compensate 59/30/10 imbalance) ----
class_weights = None
if CFG.USE_CLASS_WEIGHTS:
    counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    weights = counts.sum() / (CFG.NUM_CLASSES * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=CFG.DEVICE)
    print(f"Class counts : {counts.astype(int).tolist()}  (Neg/Neu/Pos)")
    print(f"Class weights: {np.round(weights, 3).tolist()}")

criterion = EvidentialLoss(
    num_classes=CFG.NUM_CLASSES,
    annealing_epochs=CFG.ANNEALING_EPOCHS,
    class_weights=class_weights
)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG.LR,
    weight_decay=CFG.WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.SCHED_TMAX)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "train_f1": [], "val_f1": [],
    "train_f1_macro": [], "val_f1_macro": [],
    "train_conflict": [], "val_conflict": []
}

best_score = 0.0
best_model_state = None

for epoch in range(1, CFG.EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    train_conflict_sum = 0.0
    train_preds, train_labels = [], []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(CFG.DEVICE)
        attention_mask = batch["attention_mask"].to(CFG.DEVICE)
        images = batch["image"].to(CFG.DEVICE)
        labels = batch["label"].to(CFG.DEVICE)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask, images)

        # Multi-task supervised loss on 3 branches
        loss_t = criterion(out["alpha_t"], labels, epoch)
        loss_v = criterion(out["alpha_v"], labels, epoch)
        loss_c = criterion(out["alpha_c"], labels, epoch)
        L_sup = loss_t + loss_v + loss_c

        # Fused-opinion supervision (final decision is optimized directly)
        alpha_f = opinion_to_dirichlet(out["b_fusion"], out["u_fusion"], CFG.NUM_CLASSES, CFG.U_MIN)
        L_fused = criterion(alpha_f, labels, epoch)

        # Semantic conflict loss
        L_con = semantic_conflict_loss(out["alpha_t"], out["alpha_v"], CFG.NUM_CLASSES)

        loss = L_sup + CFG.LAMBDA_FUSED * L_fused + CFG.GAMMA * L_con

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP)
        optimizer.step()

        train_loss += loss.item()
        train_conflict_sum += out["K_tv"].mean().item()

        preds = torch.argmax(out["p_final"], dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "K_tv": f"{out['K_tv'].mean().item():.3f}"})

    scheduler.step()

    # ---- VALIDATE ----
    (same structure, compute val losses)
    
    # metrics: acc, weighted f1, macro f1
    ...
    score = val_f1_macro if CFG.SELECT_METRIC == "macro_f1" else val_f1
    if score > best_score:
        best_score = score
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  \u2705 New best model! (Val {CFG.SELECT_METRIC}: {best_score:.4f})")

print(f"\n{'='*60}")
print(f"Training complete! Best Val {CFG.SELECT_METRIC}: {best_score:.4f}")
print(f"{'='*60}")
```

Val loop computes the same losses (with epoch for annealing, consistent with original).

Now cell 13:

```python
# ============================================================
# TEST SET EVALUATION
# ============================================================

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("\u2705 Best model loaded for evaluation.")

model.eval()
all_preds = []
all_labels = []
all_uncertainties = []
all_conflicts = []
all_route_a = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        input_ids = batch["input_ids"].to(CFG.DEVICE)
        attention_mask = batch["attention_mask"].to(CFG.DEVICE)
        images = batch["image"].to(CFG.DEVICE)
        labels = batch["label"].to(CFG.DEVICE)

        out = model(input_ids, attention_mask, images)

        preds = torch.argmax(out["p_final"], dim=1)
        uncertainty = out["u_fusion"].squeeze(1)
        conflict = out["K_tv"].squeeze(1)
        route_a = out["route_a_mask"].squeeze(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_uncertainties.extend(uncertainty.cpu().numpy())
        all_conflicts.extend(conflict.cpu().numpy())
        all_route_a.extend(route_a.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_uncertainties = np.array(all_uncertainties)
all_conflicts = np.array(all_conflicts)
all_route_a = np.array(all_route_a)

# ============================================================
# METRICS
# ============================================================
test_acc = accuracy_score(all_labels, all_preds)
test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")
test_f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_per_class = f1_score(all_labels, all_preds, average=None)
uce_score = compute_uce(all_preds, all_labels, all_uncertainties, num_bins=CFG.UCE_BINS)

print(f"\n{'='*60}")
print(f"TEST SET RESULTS (ADEF Co-Attention v2)")
print(f"{'='*60}")
print(f"Accuracy:          {test_acc:.4f}")
print(f"F1 (Weighted):     {test_f1_weighted:.4f}")
print(f"F1 (Macro):        {test_f1_macro:.4f}")
print(f"F1 (Negative):     {f1_per_class[0]:.4f}")
print(f"F1 (Neutral):      {f1_per_class[1]:.4f}   <- v1 was 0.0000 (collapsed)")
print(f"F1 (Positive):     {f1_per_class[2]:.4f}")
print(f"UCE:               {uce_score:.4f}")
print(f"{'='*60}")

print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]))

# ADEF routing statistics + K_tv quantiles for tau calibration
route_a_total = np.mean(all_route_a) * 100
quantiles = np.percentile(all_conflicts, [50, 80, 90, 95, 99])
print(f"\n{'='*60}")
print(f"ADEF ROUTING STATISTICS (tau = {CFG.TAU})")
print(f"{'='*60}")
print(f"Route A (Normal DS Fusion):     {route_a_total:.1f}% of samples")
print(f"Route B (Conflict-Aware):       {100 - route_a_total:.1f}% of samples")
print(f"Mean conflict K_tv:             {all_conflicts.mean():.4f} +/- {all_conflicts.std():.4f}")
print(f"K_tv quantiles p50/p80/p90/p95/p99:")
print(f"  {np.round(quantiles, 4).tolist()}")
print(f"  -> tau calibration guide: p80 = {quantiles[1]:.4f}")

# Uncertainty analysis
correct_mask = all_preds == all_labels
print(f"\n{'='*60}")
print(f"UNCERTAINTY ANALYSIS")
print(f"{'='*60}")
if correct_mask.sum() > 0:
    print(f"Mean uncertainty (correct):   {all_uncertainties[correct_mask].mean():.4f} +/- {all_uncertainties[correct_mask].std():.4f}")
if (~correct_mask).sum() > 0:
    print(f"Mean uncertainty (incorrect): {all_uncertainties[~correct_mask].mean():.4f} +/- {all_uncertainties[~correct_mask].std():.4f}")
print(f"{'='*60}")
```

Cell 14 viz: modify F1 plot to include macro, threshold lines CFG.TAU:

Plot 3 (F1 curve): 
```python
axes[0, 2].plot(history["train_f1_macro"], label="Train Macro-F1", marker="o", markersize=3)
axes[0, 2].plot(history["val_f1_macro"], label="Val Macro-F1", marker="s", markersize=3)
axes[0, 2].plot(history["val_f1"], label="Val Weighted-F1", linestyle="--", alpha=0.6)
```

Plot 4: `axes[0, 3].axhline(y=CFG.TAU, ...)` label=f"Threshold (tau={CFG.TAU})"

Plot 8: use CFG.TAU for the split and axvline:
```python
axes[1, 3].hist(all_conflicts[all_conflicts <= CFG.TAU], ..., label=f"Route A (n={(all_conflicts <= CFG.TAU).sum()})", ...)
axes[1, 3].hist(all_conflicts[all_conflicts > CFG.TAU], ..., label=f"Route B (n={(all_conflicts > CFG.TAU).sum()})", ...)
axes[1, 3].axvline(x=CFG.TAU, color="red", linestyle="--", alpha=0.7, label=f"tau={CFG.TAU}")
```

Also cell 14 title mention "v2".

Cell 6 note: F.softmax — F is imported in cell 1 (torch.nn.functional as F). Good.

Cell 5 — TextEncoder masked mean pool with clamp — good. ImageEncoder: mean over patches.

One more consideration in cell 9: docstring update.

Also potential edge: `opinion_to_dirichlet` in cell 10 defined after EvidentialLoss. Fine.

EvidentialLoss with register_buffer: when class_weights None, set self.class_weights = None (can't register None buffer... Actually register_buffer with None is allowed in PyTorch >= 1.5 — yes, register_buffer('x', None) is allowed). But simpler:

```python
def __init__(self, num_classes=3, annealing_epochs=10, class_weights=None):
    super().__init__()
    self.num_classes = num_classes
    self.annealing_epochs = annealing_epochs
    self.register_buffer("class_weights", class_weights if class_weights is not None else torch.tensor([]))
```

Hmm, empty tensor buffer — then check `if self.class_weights.numel() > 0`. Or keep plain attribute (class_weights passed already on device; it's a leaf tensor, no grads needed — plain attribute is fine and stays on device since created on device). Simplest: plain attribute.

```python
self.class_weights = class_weights  # tensor [K] on device, or None
```

Fine.

bayes_risk_loss signature: (alpha, y_onehot, sample_weight=None).

Now, cell 8 route_a_mask return — 5-tuple. Cell 9 unpacks 5 values. OK.

Let me also double check cell 10's kl_divergence_reg — keep original implementation verbatim.

Now compose the patch script file. It'll be long (~650 lines). Write it to temp dir: `C:\Users\ERLANG~1\AppData\Local\Temp\opencode\patch_adef_nb.py`.

Wait — temp dir: `C:\Users\ERLANG~1\AppData\Local\Temp\opencode`. Use that path.

Then run with base anaconda python (3.13): `& "C:\Users\erlanggadewasakti\anaconda3\python.exe" "C:\Users\ERLANG~1\AppData\Local\Temp\opencode\patch_adef_nb.py"`.

Then validation script: load notebook json, for each code cell compile(source, '<cell i>', 'exec') with ast. Note: compile of cells containing top-level `await`? None. Encoding: utf-8. The `\u2705` inside string literals is fine for compile.

Also validate JSON parses and cell count == 15.

Let me write the patch script now. I need to be extremely careful with the raw strings — the content includes `\u2705` (fine in raw), `{` and `}` f-strings (fine), no `'''` sequences. Check cell sources for occurrences of `'''` — none. But they do contain `"` and `'` quotes — fine inside triple-single-quotes as long as not three in a row.

One more: cell 14 has `f"tau={CFG.TAU}"` etc. fine.

Also important: in the patch script itself, I must avoid the temp path with the `~` short name... Test-Path earlier confirmed `C:\Users\ERLANG~1\AppData\Local\Temp\opencode` exists (pre-created per system prompt). Writing there with the Write tool — the Write tool needs absolute path; use `C:\Users\erlanggadewasakti\AppData\Local\Temp\opencode\patch_adef_nb.py` (long form should also work).

Let me write it.

Plan doc rewritten. Now the notebook patch — I'll write a Python patch script (JSON-level, no ML deps needed) with all 10 replacement cells:

**Tool: write**

**Input:**
```json
{
  "content": "# Patch adef_co_attention.ipynb: replaces sources of cells 2,3,5,6,8,9,10,12,13,14\n# and clears all outputs. JSON-only operation (no ML dependencies required).\nimport json\nimport io\n\nNB = r\"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\"\n\ncells_src = {}\n\n# ============================================================\n# CELL 2 - CONFIGURATION (all hyperparameters, incl. TAU)\n# ============================================================\ncells_src[2] = r'''# ============================================================\n# CONFIGURATION (single source of truth for ALL hyperparameters)\n# ============================================================\n\nclass CFG:\n\n    # =========================\n    # PATH\n    # =========================\n    ROOT_DIR = r\"D:/MVSA_SINGLE\"\n    DATA_DIR = r\"D:/MVSA_SINGLE/data\"\n    LABEL_PATH = r\"D:/MVSA_SINGLE/labelResultAllFinal.txt\"\n\n    # =========================\n    # DEVICE\n    # =========================\n    DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n\n    # =========================\n    # DATA\n    # =========================\n    # True  -> drop contradictory text-image pairs (identical to the 4\n    #          baseline notebooks, fair comparison).\n    # False -> keep them (ablation: gives ADEF Route B real conflict cases).\n    FILTER_CONFLICT_PAIRS = True\n\n    # =========================\n    # OPTIMIZATION HYPERPARAMETERS\n    # =========================\n    BATCH_SIZE = 16\n    EPOCHS = 30\n    LR = 1e-4              # heads/co-attn are trained from scratch (backbones frozen)\n                           # -> needs >= 1e-4. v1 used 2e-5 -> undertrained heads.\n    WEIGHT_DECAY = 1e-4\n    GRAD_CLIP = 1.0\n    SCHED_TMAX = 30        # CosineAnnealingLR period (usually == EPOCHS)\n    DROPOUT = 0.3\n    SEED = 42\n\n    # =========================\n    # ARCHITECTURE HYPERPARAMETERS\n    # =========================\n    MAX_LEN = 150\n    D_BERT = 768\n    D_CNN = 1024\n    D_PROJ = 512\n    NUM_CLASSES = 3\n\n    # =========================\n    # EDL / ADEF HYPERPARAMETERS\n    # =========================\n    ANNEALING_EPOCHS = 10    # KL term reaches full strength at this epoch\n    TAU = 0.1                # ADEF conflict threshold. v1 hardcoded 0.5 while\n                             # mean K_tv ~ 0.014 -> Route B never fired.\n                             # Calibrate via K_tv quantile report (p80) on val/test.\n    GAMMA = 1.0              # weight of semantic conflict loss L_con\n    LAMBDA_FUSED = 1.0       # weight of fused-opinion supervision L_fused\n    USE_CLASS_WEIGHTS = True # class-weighted evidential loss (fixes neutral collapse)\n    U_MIN = 0.05             # lower clamp of u_fusion when rebuilding alpha_fusion\n\n    # =========================\n    # MODEL SELECTION / EVALUATION\n    # =========================\n    SELECT_METRIC = \"macro_f1\"  # best checkpoint by MACRO F1 (v1 used weighted F1\n                                # -> majority-biased, selected a 0%-neutral-recall model)\n    UCE_BINS = 10\n\n    # =========================\n    # PRETRAINED MODELS\n    # =========================\n    TEXT_MODEL = \"roberta-base\"\n    IMAGE_MODEL = \"densenet121\"\n\nprint(f\"\\u2705 Configuration loaded. Device: {CFG.DEVICE}\")'''\n\n# ============================================================\n# CELL 3 - LOAD DATASET (filter behind CFG flag)\n# ============================================================\ncells_src[3] = r'''import pandas as pd\nimport os\n# ============================================================\n# LOAD DATASET\n# ============================================================\n\ndf = pd.read_csv(CFG.LABEL_PATH, header=0, sep=\",\")\ndf.columns = [\"id\", \"text_label\", \"image_label\", \"final_label\"]\n\n# Filter contradictory text-image pairs (text+ / image- and vice versa).\n# NOTE: these ARE the cross-modal conflict cases ADEF Route B was designed for.\n# Default True keeps the dataset identical to the 4 baseline notebooks\n# (fair comparison). Set CFG.FILTER_CONFLICT_PAIRS = False for ablation.\ndef is_valid(row):\n\n    if row[\"text_label\"] == \"positive\" and row[\"image_label\"] == \"negative\":\n        return False\n\n    if row[\"text_label\"] == \"negative\" and row[\"image_label\"] == \"positive\":\n        return False\n\n    return True\n\nif CFG.FILTER_CONFLICT_PAIRS:\n    df = df[df.apply(is_valid, axis=1)]\n    df = df.reset_index(drop=True)\n\nprint(f\"Dataset size after filtering: {len(df)}  (FILTER_CONFLICT_PAIRS={CFG.FILTER_CONFLICT_PAIRS})\")\n\nlabel_map = {\n    \"negative\": 0,\n    \"neutral\": 1,\n    \"positive\": 2\n}\n\nid2label = {\n    0: \"negative\",\n    1: \"neutral\",\n    2: \"positive\"\n}\n\ndf[\"label\"] = df[\"final_label\"].map(label_map)\n\n# Load text file dengan better error handling\ndef load_text(sample_id):\n    path = os.path.join(CFG.DATA_DIR, f\"{sample_id}.txt\")\n\n    encodings = [\"utf-8\", \"latin-1\", \"cp1252\", \"iso-8859-1\"]\n\n    for encoding in encodings:\n        try:\n            with open(path, \"r\", encoding=encoding) as f:\n                text = f.read().strip()\n                if text:  # Jika text berhasil dibaca dan tidak kosong\n                    return text\n        except FileNotFoundError:\n            continue\n        except Exception as e:\n            continue\n\n    # Jika semua encoding gagal atau file tidak ada\n    return \"\"\n\ndf[\"text\"] = df[\"id\"].apply(load_text)\n\n# Hitung empty text\nempty_text_count = (df[\"text\"] == \"\").sum()\nprint(f\"\\n{'='*60}\")\nprint(f\"PREPROCESSING STATISTICS:\")\nprint(f\"{'='*60}\")\nprint(f\"Total samples: {len(df)}\")\nprint(f\"Samples with EMPTY text: {empty_text_count}\")\nprint(f\"Samples with VALID text: {len(df) - empty_text_count}\")\nprint(f\"Percentage of empty text: {(empty_text_count/len(df)*100):.2f}%\")\nprint(f\"{'='*60}\\n\")\n\n# image path\ndf[\"image_path\"] = df[\"id\"].apply(\n    lambda x: os.path.join(CFG.DATA_DIR, f\"{x}.jpg\")\n)\n\ndf.head()'''\n\n# ============================================================\n# CELL 5 - FEATURE EXTRACTORS (sequence-level)\n# ============================================================\ncells_src[5] = r'''# ============================================================\n# FEATURE EXTRACTORS (Sequence-Level for Co-Attention)\n# ============================================================\n\n# RoBERTa text encoder -> token-level features + masked mean-pooled vector.\n# v1 pooled to a single vector BEFORE attention, degenerating the attention\n# matrix to a scalar. v2 keeps token-level features for fine-grained alignment.\n# Input:  input_ids [B, L_t], attention_mask [B, L_t]\n# Output: H_t  [B, L_t, d_proj] (token features for co-attention)\n#         h_t  [B, d_proj]      (pooled vector for unimodal ENN head)\n#         attention_mask [B, L_t]\nclass TextEncoder(nn.Module):\n\n    def __init__(self, d_bert=768, d_proj=512):\n        super().__init__()\n        self.roberta = RobertaModel.from_pretrained(CFG.TEXT_MODEL)\n        # Freeze RoBERTa parameters (fair comparison with baseline notebooks)\n        for param in self.roberta.parameters():\n            param.requires_grad = False\n\n        self.projection = nn.Sequential(\n            nn.Linear(d_bert, d_proj),\n            nn.ReLU(),\n            nn.LayerNorm(d_proj)\n        )\n\n    def forward(self, input_ids, attention_mask):\n        with torch.no_grad():\n            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)\n        H_t = self.projection(outputs.last_hidden_state)  # [B, L_t, d_proj]\n\n        # Masked mean-pooling -> pooled text vector\n        mask = attention_mask.unsqueeze(-1).float()       # [B, L_t, 1]\n        h_t = (H_t * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-10)  # [B, d_proj]\n        return H_t, h_t, attention_mask\n\n\n# DenseNet-121 image encoder -> spatial (patch) features + pooled vector.\n# Input:  image [B, 3, 224, 224]\n# Output: H_v [B, N_v, d_proj] (N_v = 49 patch features for co-attention)\n#         h_v [B, d_proj]      (pooled vector for unimodal ENN head)\nclass ImageEncoder(nn.Module):\n\n    def __init__(self, d_cnn=1024, d_proj=512):\n        super().__init__()\n        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)\n        self.features = densenet.features\n        # Freeze DenseNet parameters\n        for param in self.features.parameters():\n            param.requires_grad = False\n\n        self.projection = nn.Sequential(\n            nn.Linear(d_cnn, d_proj),\n            nn.ReLU(),\n            nn.LayerNorm(d_proj)\n        )\n\n    def forward(self, x):\n        with torch.no_grad():\n            features = self.features(x)  # [B, 1024, 7, 7]\n        B, C, H, W = features.shape\n\n        # Spatial features: [B, N_v, 1024] -> project\n        H_v = features.view(B, C, H * W).permute(0, 2, 1)  # [B, N_v, 1024]\n        H_v = self.projection(H_v)                          # [B, N_v, d_proj]\n        h_v = H_v.mean(dim=1)                               # [B, d_proj]\n        return H_v, h_v\n\n\nprint(\"\\u2705 TextEncoder (sequence) & ImageEncoder (spatial) defined.\")'''\n\n# ============================================================\n# CELL 6 - BIDIRECTIONAL CO-ATTENTION (sequence-level, thesis formula)\n# ============================================================\ncells_src[6] = r'''# ============================================================\n# BIDIRECTIONAL CO-ATTENTION MODULE (Sequence-Level)\n# ============================================================\n\n# Fine-grained bidirectional alignment between text tokens and image patches,\n# faithful to the thesis formulas but at sequence level (v1 applied them to\n# pooled vectors, degenerating A to a [B,1,1] sigmoid scalar):\n#   S    = H_t W H_v^T / sqrt(d)                 -> [B, L_t, N_v]\n#   A_tv = softmax(S, dim=patches)               (each token -> over patches)\n#   A_vt = softmax(S^T, dim=tokens, masked)      (each patch -> over tokens)\n#   h_c  = MLP( Concat( pool(A_vt H_t), pool(A_tv H_v) ) )\n#\n# Input:  H_t [B, L_t, d_proj], H_v [B, N_v, d_proj], text_mask [B, L_t]\n# Output: h_c [B, d_proj]\nclass BiCoAttention(nn.Module):\n\n    def __init__(self, d_proj=512, dropout=0.3):\n        super().__init__()\n        self.d_proj = d_proj\n\n        # Projection matrix W for attention scores\n        self.W_attn = nn.Linear(d_proj, d_proj, bias=False)\n\n        # Project concatenated representation [2*d_proj] back to [d_proj]\n        self.fusion_proj = nn.Sequential(\n            nn.Linear(2 * d_proj, d_proj),\n            nn.ReLU(),\n            nn.LayerNorm(d_proj),\n            nn.Dropout(dropout)\n        )\n\n    def forward(self, H_t, H_v, text_mask):\n        # Affinity matrix: S = H_t W H_v^T / sqrt(d)\n        S = torch.bmm(self.W_attn(H_t), H_v.transpose(1, 2)) / (self.d_proj ** 0.5)  # [B, L_t, N_v]\n\n        # --- Text-guided visual attention (each token attends over patches) ---\n        A_tv = F.softmax(S, dim=-1)                          # [B, L_t, N_v]\n        v_att_tokens = torch.bmm(A_tv, H_v)                  # [B, L_t, d]\n        mask = text_mask.unsqueeze(-1).float()               # [B, L_t, 1]\n        v_att = (v_att_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-10)  # [B, d]\n\n        # --- Visual-guided textual attention (each patch attends over tokens) ---\n        pad = (text_mask == 0).unsqueeze(1)                  # [B, 1, L_t]\n        S_T = S.transpose(1, 2).masked_fill(pad, -1e9)       # [B, N_v, L_t]\n        A_vt = F.softmax(S_T, dim=-1)                        # [B, N_v, L_t]\n        t_att = torch.bmm(A_vt, H_t).mean(dim=1)             # [B, N_v, d] -> [B, d]\n\n        # --- Concatenate and project ---\n        h_c = self.fusion_proj(torch.cat([t_att, v_att], dim=1))  # [B, d]\n        return h_c\n\n\nprint(\"\\u2705 BiCoAttention (sequence-level) module defined.\")'''\n\n# ============================================================\n# CELL 8 - ADEF MODULE (tau from config, numeric guards)\n# ============================================================\ncells_src[8] = r'''# ============================================================\n# ADAPTIVE EVIDENTIAL FUSION (ADEF) MODULE\n# ============================================================\n\n# ADEF performs dynamic routing based on conflict level K_tv between\n# text and image modalities.\n#\n# Route A (K_tv <= tau): Normal Dempster-Shafer fusion in 2 stages\n#   Stage 1: fuse text + image -> (b_tv, u_tv)\n#   Stage 2: fuse (b_tv, u_tv) + co-attention -> final\n#\n# Route B (K_tv > tau): Conflict-aware bypass\n#   Uses K_tv as dynamic lever to blend unimodal average with co-attention\n#\n# tau is a CONFIGURED hyperparameter (CFG.TAU), calibrated to the empirical\n# K_tv scale. v1 hardcoded tau=0.5 with mean K_tv ~ 0.014 -> Route B dead.\nclass ADEFModule(nn.Module):\n\n    def __init__(self, num_classes=3, tau=0.1):\n        super().__init__()\n        self.num_classes = num_classes\n        self.tau = tau\n\n    # Dempster's Rule of Combination for two belief structures\n    # b1, u1: belief [B, K] and uncertainty [B, 1] from source 1\n    # b2, u2: belief [B, K] and uncertainty [B, 1] from source 2\n    # Returns: b_fused [B, K], u_fused [B, 1], conflict C [B, 1]\n    def dempster_combine(self, b1, u1, b2, u2):\n        eps = 1e-8\n\n        # Conflict: C = sum_{i!=j} b1_i * b2_j = (sum b1)(sum b2) - sum(b1*b2)\n        b1_sum = torch.sum(b1, dim=1, keepdim=True)\n        b2_sum = torch.sum(b2, dim=1, keepdim=True)\n        C = b1_sum * b2_sum - torch.sum(b1 * b2, dim=1, keepdim=True)\n\n        # Normalization: 1 / (1 - C), guarded against total conflict\n        norm = 1.0 / torch.clamp(1.0 - C, min=eps)\n\n        # Fused belief and uncertainty\n        b_fused = norm * (b1 * b2 + b1 * u2 + b2 * u1)  # [B, K]\n        u_fused = norm * (u1 * u2)                      # [B, 1]\n\n        return b_fused, u_fused, C\n\n    def forward(self, b_t, u_t, b_v, u_v, b_c, u_c):\n        # b_t, b_v, b_c: [B, K] - belief masses\n        # u_t, u_v, u_c: [B, 1] - uncertainty masses\n        eps = 1e-8\n        K = self.num_classes\n\n        # ---- Compute conflict K_tv between text and image ----\n        # K_tv = sum_{i!=j} b_t_i * b_v_j\n        b_t_sum = torch.sum(b_t, dim=1, keepdim=True)\n        b_v_sum = torch.sum(b_v, dim=1, keepdim=True)\n        K_tv = b_t_sum * b_v_sum - torch.sum(b_t * b_v, dim=1, keepdim=True)  # [B, 1]\n\n        # ---- Dynamic Routing (per-sample) ----\n        route_a_mask = (K_tv <= self.tau).float()  # [B, 1], 1 = Route A, 0 = Route B\n\n        # ---- ROUTE A: Normal Dempster-Shafer Fusion (2-stage) ----\n        b_tv_a, u_tv_a, _ = self.dempster_combine(b_t, u_t, b_v, u_v)\n        b_final_a, u_final_a, _ = self.dempster_combine(b_tv_a, u_tv_a, b_c, u_c)\n\n        # ---- ROUTE B: Conflict-Aware Bypass Fusion ----\n        b_avg = (b_t + b_v) / 2.0\n        b_final_b = (1.0 - K_tv) * b_avg + K_tv * b_c                 # [B, K]\n        u_final_b = 1.0 - torch.sum(b_final_b, dim=1, keepdim=True)   # [B, 1]\n        u_final_b = torch.clamp(u_final_b, min=eps)\n\n        # ---- Merge routes per sample ----\n        b_fusion = route_a_mask * b_final_a + (1.0 - route_a_mask) * b_final_b\n        u_fusion = route_a_mask * u_final_a + (1.0 - route_a_mask) * u_final_b\n        u_fusion = torch.clamp(u_fusion, min=eps, max=1.0)\n\n        # ---- Final Decision: p_i = b_i + u / M ----\n        p_final = b_fusion + u_fusion / K  # [B, K]\n\n        return p_final, b_fusion, u_fusion, K_tv, route_a_mask\n\n\nprint(\"\\u2705 ADEFModule defined.\")'''\n\n# ============================================================\n# CELL 9 - MODEL ADEFCoAttnNet\n# ============================================================\ncells_src[9] = r'''# ============================================================\n# MODEL: ADEFCoAttnNet (v2)\n# ============================================================\n\n# Adaptive Evidential Fusion with sequence-level Co-Attention Network\n# Architecture:\n#   1. TextEncoder  -> H_t [B, L_t, d], h_t [B, d]\n#   2. ImageEncoder -> H_v [B, N_v, d], h_v [B, d]\n#   3. BiCoAttention(H_t, H_v, mask) -> h_c [B, d]\n#   4. 3x ENNHead(h_t, h_v, h_c) -> alpha_t, alpha_v, alpha_c\n#   5. Subjective Logic -> belief & uncertainty per branch\n#   6. ADEF(tau=CFG.TAU) -> dynamic routing -> final prediction\nclass ADEFCoAttnNet(nn.Module):\n\n    def __init__(self, d_proj=512, num_classes=3, dropout=0.3, tau=0.1):\n        super().__init__()\n        # Feature extractors (sequence-level)\n        self.text_encoder = TextEncoder(d_bert=CFG.D_BERT, d_proj=d_proj)\n        self.image_encoder = ImageEncoder(d_cnn=CFG.D_CNN, d_proj=d_proj)\n\n        # Co-Attention module\n        self.co_attention = BiCoAttention(d_proj=d_proj, dropout=dropout)\n\n        # 3 independent ENN heads\n        self.enn_text = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)\n        self.enn_image = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)\n        self.enn_coattn = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)\n\n        # ADEF fusion module\n        self.adef = ADEFModule(num_classes=num_classes, tau=tau)\n\n        self.num_classes = num_classes\n\n    def forward(self, input_ids, attention_mask, image):\n        # 1. Feature extraction (sequence-level + pooled)\n        H_t, h_t, text_mask = self.text_encoder(input_ids, attention_mask)\n        H_v, h_v = self.image_encoder(image)\n\n        # 2. Co-Attention (token <-> patch alignment)\n        h_c = self.co_attention(H_t, H_v, text_mask)  # [B, d]\n\n        # 3. ENN Heads -> Dirichlet parameters\n        alpha_t = self.enn_text(h_t)    # [B, K]\n        alpha_v = self.enn_image(h_v)   # [B, K]\n        alpha_c = self.enn_coattn(h_c)  # [B, K]\n\n        # 4. Subjective Logic: belief & uncertainty\n        b_t, u_t, S_t = compute_belief_uncertainty(alpha_t, self.num_classes)\n        b_v, u_v, S_v = compute_belief_uncertainty(alpha_v, self.num_classes)\n        b_c, u_c, S_c = compute_belief_uncertainty(alpha_c, self.num_classes)\n\n        # 5. ADEF fusion\n        p_final, b_fusion, u_fusion, K_tv, route_a_mask = self.adef(\n            b_t, u_t, b_v, u_v, b_c, u_c\n        )\n\n        return {\n            \"alpha_t\": alpha_t,\n            \"alpha_v\": alpha_v,\n            \"alpha_c\": alpha_c,\n            \"b_t\": b_t, \"u_t\": u_t,\n            \"b_v\": b_v, \"u_v\": u_v,\n            \"b_c\": b_c, \"u_c\": u_c,\n            \"p_final\": p_final,\n            \"b_fusion\": b_fusion,\n            \"u_fusion\": u_fusion,\n            \"K_tv\": K_tv,\n            \"route_a_mask\": route_a_mask\n        }\n\n\nmodel = ADEFCoAttnNet(\n    d_proj=CFG.D_PROJ,\n    num_classes=CFG.NUM_CLASSES,\n    dropout=CFG.DROPOUT,\n    tau=CFG.TAU\n).to(CFG.DEVICE)\n\n# Count parameters\ntotal_params = sum(p.numel() for p in model.parameters())\ntrainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\nprint(f\"\\u2705 ADEFCoAttnNet (v2) model created.\")\nprint(f\"Total parameters: {total_params:,}\")\nprint(f\"Trainable parameters: {trainable_params:,}\")\nprint(f\"Frozen parameters: {total_params - trainable_params:,}\")'''\n\n# ============================================================\n# CELL 10 - LOSSES (class-weighted evidential + fused-opinion supervision)\n# ============================================================\ncells_src[10] = r'''# ============================================================\n# BLENDED LOSS FUNCTION (v2)\n# L_overall = L_sup + LAMBDA_FUSED * L_fused + GAMMA * L_con\n# ============================================================\n\n# Evidential Loss: class-weighted Bayes Risk (Sum of Squares) + KL Divergence.\n# L(alpha) = w_y * L_err(alpha) + lambda_t * L_KL(alpha)\n#\n# v2 changes:\n#  - per-sample class weight w_y (inverse class frequency) on L_err:\n#    without it, the 59/30/10 imbalance collapses the Neutral class to 0% recall.\nclass EvidentialLoss(nn.Module):\n\n    def __init__(self, num_classes=3, annealing_epochs=10, class_weights=None):\n        super().__init__()\n        self.num_classes = num_classes\n        self.annealing_epochs = annealing_epochs\n        self.class_weights = class_weights  # tensor [K] on device, or None\n\n    # Error Term (Sum of Squares Loss / Bayes Risk):\n    # L_err = sum_j (y_j - p_j)^2 + p_j(1 - p_j) / (S + 1)\n    def bayes_risk_loss(self, alpha, y_onehot, sample_weight=None):\n        alpha = torch.clamp(alpha, min=1e-10)\n        S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]\n        p_hat = alpha / S                           # [B, K]\n\n        # Sum of Squares error\n        err = torch.sum((y_onehot - p_hat) ** 2, dim=1)   # [B]\n\n        # Variance term (epistemic uncertainty)\n        var = torch.sum(p_hat * (1.0 - p_hat) / (S + 1.0), dim=1)  # [B]\n\n        loss = err + var  # [B]\n        if sample_weight is not None:\n            loss = loss * sample_weight\n        return loss.mean()\n\n    # KL Divergence regularization: KL(Dir(alpha_tilde) || Dir(1,...,1))\n    # alpha_tilde = y + (1 - y) * alpha  (remove correct-class evidence)\n    def kl_divergence_reg(self, alpha, y_onehot):\n        alpha = torch.clamp(alpha, min=1e-10)\n        K = self.num_classes\n\n        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha\n        alpha_tilde = torch.clamp(alpha_tilde, min=1e-10)\n        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)\n\n        kl = (\n            torch.lgamma(S_tilde)\n            - torch.lgamma(torch.tensor(float(K), device=alpha.device))\n            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)\n            + torch.sum(\n                (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),\n                dim=1, keepdim=True\n            )\n        )\n        return kl.mean()\n\n    # Forward: L(alpha) = w_y * L_err + lambda_t * L_KL\n    def forward(self, alpha, labels, epoch):\n        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()\n        lambda_t = min(1.0, epoch / max(self.annealing_epochs, 1))\n\n        sample_weight = None\n        if self.class_weights is not None:\n            sample_weight = self.class_weights[labels]  # [B]\n\n        loss_err = self.bayes_risk_loss(alpha, y_onehot, sample_weight)\n        loss_kl = self.kl_divergence_reg(alpha, y_onehot)\n\n        return loss_err + lambda_t * loss_kl\n\n\n# Reconstruct an equivalent Dirichlet distribution from a fused\n# Subjective-Logic opinion (b_fusion, u_fusion):\n#   S_fusion = M / clamp(u_fusion, u_min, 1)\n#   alpha_fusion = b_fusion * S_fusion + 1\n# Enables DIRECT supervision of the final fused decision (v1 left p_final\n# outside the loss, so the decision was only optimized indirectly).\ndef opinion_to_dirichlet(b, u, num_classes=3, u_min=0.05):\n    u_c = torch.clamp(u, min=u_min, max=1.0)\n    S = num_classes / u_c\n    return b * S + 1.0\n\n\n# Semantic Conflict Loss: L_con = d_PD * d_CC\n# d_PD = 0.5 * (1 - u_t)(1 - u_v)\n# d_CC = sum_i |p_t_i - p_v_i|\n# Guides the encoder to detect emotional incongruence (sarcasm)\ndef semantic_conflict_loss(alpha_t, alpha_v, num_classes=3, eps=1e-8):\n    S_t = torch.sum(alpha_t, dim=1, keepdim=True)\n    S_v = torch.sum(alpha_v, dim=1, keepdim=True)\n\n    p_t = alpha_t / (S_t + eps)          # [B, K]\n    p_v = alpha_v / (S_v + eps)          # [B, K]\n\n    u_t = num_classes / (S_t + eps)      # [B, 1]\n    u_v = num_classes / (S_v + eps)      # [B, 1]\n\n    # d_PD: confidence product - high when both modalities are confident\n    d_PD = 0.5 * (1.0 - u_t) * (1.0 - u_v)  # [B, 1]\n\n    # d_CC: cross-modal class disagreement\n    d_CC = torch.sum(torch.abs(p_t - p_v), dim=1, keepdim=True)  # [B, 1]\n\n    # L_con = d_PD * d_CC\n    loss_con = (d_PD * d_CC).mean()\n    return loss_con\n\n\nprint(\"\\u2705 EvidentialLoss (class-weighted) & opinion_to_dirichlet & SemanticConflictLoss defined.\")'''\n\n# ============================================================\n# CELL 12 - TRAINING LOOP\n# ============================================================\ncells_src[12] = r'''# ============================================================\n# TRAINING & VALIDATION (ADEF Blended Loss v2)\n# L_overall = L_sup + LAMBDA_FUSED * L_fused + GAMMA * L_con\n# ============================================================\n\n# ---- Class weights from TRAIN split (compensate 59/30/10 imbalance) ----\nclass_weights = None\nif CFG.USE_CLASS_WEIGHTS:\n    counts = train_df[\"label\"].value_counts().sort_index().values.astype(np.float32)\n    weights = counts.sum() / (CFG.NUM_CLASSES * counts)  # w_c = N / (M * n_c)\n    class_weights = torch.tensor(weights, dtype=torch.float32, device=CFG.DEVICE)\n    print(f\"Class counts  (Neg/Neu/Pos): {counts.astype(int).tolist()}\")\n    print(f\"Class weights (Neg/Neu/Pos): {np.round(weights, 3).tolist()}\")\n\ncriterion = EvidentialLoss(\n    num_classes=CFG.NUM_CLASSES,\n    annealing_epochs=CFG.ANNEALING_EPOCHS,\n    class_weights=class_weights\n)\n\noptimizer = optim.AdamW(\n    filter(lambda p: p.requires_grad, model.parameters()),\n    lr=CFG.LR,\n    weight_decay=CFG.WEIGHT_DECAY\n)\nscheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.SCHED_TMAX)\n\nhistory = {\n    \"train_loss\": [], \"val_loss\": [],\n    \"train_acc\": [], \"val_acc\": [],\n    \"train_f1\": [], \"val_f1\": [],\n    \"train_f1_macro\": [], \"val_f1_macro\": [],\n    \"train_conflict\": [], \"val_conflict\": []\n}\n\nbest_score = 0.0\nbest_model_state = None\n\nfor epoch in range(1, CFG.EPOCHS + 1):\n    # ---- TRAIN ----\n    model.train()\n    train_loss = 0.0\n    train_conflict_sum = 0.0\n    train_preds, train_labels = [], []\n\n    pbar = tqdm(train_loader, desc=f\"Epoch {epoch}/{CFG.EPOCHS} [Train]\")\n    for batch in pbar:\n        input_ids = batch[\"input_ids\"].to(CFG.DEVICE)\n        attention_mask = batch[\"attention_mask\"].to(CFG.DEVICE)\n        images = batch[\"image\"].to(CFG.DEVICE)\n        labels = batch[\"label\"].to(CFG.DEVICE)\n\n        optimizer.zero_grad()\n        out = model(input_ids, attention_mask, images)\n\n        # Multi-task supervised loss: L_sup = L(alpha_t) + L(alpha_v) + L(alpha_c)\n        loss_t = criterion(out[\"alpha_t\"], labels, epoch)\n        loss_v = criterion(out[\"alpha_v\"], labels, epoch)\n        loss_c = criterion(out[\"alpha_c\"], labels, epoch)\n        L_sup = loss_t + loss_v + loss_c\n\n        # Fused-opinion supervision: the FINAL decision is optimized directly\n        alpha_f = opinion_to_dirichlet(out[\"b_fusion\"], out[\"u_fusion\"],\n                                       CFG.NUM_CLASSES, CFG.U_MIN)\n        L_fused = criterion(alpha_f, labels, epoch)\n\n        # Semantic conflict loss\n        L_con = semantic_conflict_loss(out[\"alpha_t\"], out[\"alpha_v\"], CFG.NUM_CLASSES)\n\n        # Total loss\n        loss = L_sup + CFG.LAMBDA_FUSED * L_fused + CFG.GAMMA * L_con\n\n        loss.backward()\n        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP)\n        optimizer.step()\n\n        train_loss += loss.item()\n        train_conflict_sum += out[\"K_tv\"].mean().item()\n\n        # Predictions from final probability\n        preds = torch.argmax(out[\"p_final\"], dim=1)\n        train_preds.extend(preds.cpu().numpy())\n        train_labels.extend(labels.cpu().numpy())\n\n        pbar.set_postfix({\"loss\": f\"{loss.item():.4f}\", \"K_tv\": f\"{out['K_tv'].mean().item():.3f}\"})\n\n    scheduler.step()\n\n    # ---- VALIDATE ----\n    model.eval()\n    val_loss = 0.0\n    val_conflict_sum = 0.0\n    val_preds, val_labels_list = [], []\n\n    with torch.no_grad():\n        for batch in tqdm(val_loader, desc=f\"Epoch {epoch}/{CFG.EPOCHS} [Val]\"):\n            input_ids = batch[\"input_ids\"].to(CFG.DEVICE)\n            attention_mask = batch[\"attention_mask\"].to(CFG.DEVICE)\n            images = batch[\"image\"].to(CFG.DEVICE)\n            labels = batch[\"label\"].to(CFG.DEVICE)\n\n            out = model(input_ids, attention_mask, images)\n\n            loss_t = criterion(out[\"alpha_t\"], labels, epoch)\n            loss_v = criterion(out[\"alpha_v\"], labels, epoch)\n            loss_c = criterion(out[\"alpha_c\"], labels, epoch)\n            L_sup = loss_t + loss_v + loss_c\n            alpha_f = opinion_to_dirichlet(out[\"b_fusion\"], out[\"u_fusion\"],\n                                           CFG.NUM_CLASSES, CFG.U_MIN)\n            L_fused = criterion(alpha_f, labels, epoch)\n            L_con = semantic_conflict_loss(out[\"alpha_t\"], out[\"alpha_v\"], CFG.NUM_CLASSES)\n            loss = L_sup + CFG.LAMBDA_FUSED * L_fused + CFG.GAMMA * L_con\n\n            val_loss += loss.item()\n            val_conflict_sum += out[\"K_tv\"].mean().item()\n\n            preds = torch.argmax(out[\"p_final\"], dim=1)\n            val_preds.extend(preds.cpu().numpy())\n            val_labels_list.extend(labels.cpu().numpy())\n\n    # Epoch metrics\n    avg_train_loss = train_loss / len(train_loader)\n    avg_val_loss = val_loss / len(val_loader)\n    avg_train_conflict = train_conflict_sum / len(train_loader)\n    avg_val_conflict = val_conflict_sum / len(val_loader)\n    train_acc = accuracy_score(train_labels, train_preds)\n    val_acc = accuracy_score(val_labels_list, val_preds)\n    train_f1 = f1_score(train_labels, train_preds, average=\"weighted\")\n    val_f1 = f1_score(val_labels_list, val_preds, average=\"weighted\")\n    train_f1_macro = f1_score(train_labels, train_preds, average=\"macro\")\n    val_f1_macro = f1_score(val_labels_list, val_preds, average=\"macro\")\n\n    history[\"train_loss\"].append(avg_train_loss)\n    history[\"val_loss\"].append(avg_val_loss)\n    history[\"train_acc\"].append(train_acc)\n    history[\"val_acc\"].append(val_acc)\n    history[\"train_f1\"].append(train_f1)\n    history[\"val_f1\"].append(val_f1)\n    history[\"train_f1_macro\"].append(train_f1_macro)\n    history[\"val_f1_macro\"].append(val_f1_macro)\n    history[\"train_conflict\"].append(avg_train_conflict)\n    history[\"val_conflict\"].append(avg_val_conflict)\n\n    print(f\"\\nEpoch {epoch}/{CFG.EPOCHS}\")\n    print(f\"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1w: {train_f1:.4f} | F1m: {train_f1_macro:.4f} | K_tv: {avg_train_conflict:.4f}\")\n    print(f\"  Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1w: {val_f1:.4f} | F1m: {val_f1_macro:.4f} | K_tv: {avg_val_conflict:.4f}\")\n    print(f\"  LR: {scheduler.get_last_lr()[0]:.2e}\")\n\n    # ---- Best model selection: MACRO F1 (majority-agnostic) ----\n    score = val_f1_macro if CFG.SELECT_METRIC == \"macro_f1\" else val_f1\n    if score > best_score:\n        best_score = score\n        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}\n        print(f\"  \\u2705 New best model! (Val {CFG.SELECT_METRIC}: {best_score:.4f})\")\n\nprint(f\"\\n{'='*60}\")\nprint(f\"Training complete! Best Val {CFG.SELECT_METRIC}: {best_score:.4f}\")\nprint(f\"{'='*60}\")'''\n\n# ============================================================\n# CELL 13 - TEST SET EVALUATION\n# ============================================================\ncells_src[13] = r'''# ============================================================\n# TEST SET EVALUATION\n# ============================================================\n\nif best_model_state is not None:\n    model.load_state_dict(best_model_state)\n    print(\"\\u2705 Best model loaded for evaluation.\")\n\nmodel.eval()\nall_preds = []\nall_labels = []\nall_uncertainties = []\nall_conflicts = []\nall_route_a = []\n\nwith torch.no_grad():\n    for batch in tqdm(test_loader, desc=\"Evaluating on Test Set\"):\n        input_ids = batch[\"input_ids\"].to(CFG.DEVICE)\n        attention_mask = batch[\"attention_mask\"].to(CFG.DEVICE)\n        images = batch[\"image\"].to(CFG.DEVICE)\n        labels = batch[\"label\"].to(CFG.DEVICE)\n\n        out = model(input_ids, attention_mask, images)\n\n        preds = torch.argmax(out[\"p_final\"], dim=1)\n        uncertainty = out[\"u_fusion\"].squeeze(1)       # [B]\n        conflict = out[\"K_tv\"].squeeze(1)              # [B]\n        route_a = out[\"route_a_mask\"].squeeze(1)       # [B]\n\n        all_preds.extend(preds.cpu().numpy())\n        all_labels.extend(labels.cpu().numpy())\n        all_uncertainties.extend(uncertainty.cpu().numpy())\n        all_conflicts.extend(conflict.cpu().numpy())\n        all_route_a.extend(route_a.cpu().numpy())\n\nall_preds = np.array(all_preds)\nall_labels = np.array(all_labels)\nall_uncertainties = np.array(all_uncertainties)\nall_conflicts = np.array(all_conflicts)\nall_route_a = np.array(all_route_a)\n\n# ============================================================\n# METRICS\n# ============================================================\ntest_acc = accuracy_score(all_labels, all_preds)\ntest_f1_weighted = f1_score(all_labels, all_preds, average=\"weighted\")\ntest_f1_macro = f1_score(all_labels, all_preds, average=\"macro\")\nf1_per_class = f1_score(all_labels, all_preds, average=None)\nuce_score = compute_uce(all_preds, all_labels, all_uncertainties, num_bins=CFG.UCE_BINS)\n\nprint(f\"\\n{'='*60}\")\nprint(f\"TEST SET RESULTS (ADEF Co-Attention v2)\")\nprint(f\"{'='*60}\")\nprint(f\"Accuracy:          {test_acc:.4f}\")\nprint(f\"F1 (Weighted):     {test_f1_weighted:.4f}\")\nprint(f\"F1 (Macro):        {test_f1_macro:.4f}\")\nprint(f\"F1 (Negative):     {f1_per_class[0]:.4f}\")\nprint(f\"F1 (Neutral):      {f1_per_class[1]:.4f}   <- v1 collapsed to 0.0000\")\nprint(f\"F1 (Positive):     {f1_per_class[2]:.4f}\")\nprint(f\"UCE:               {uce_score:.4f}\")\nprint(f\"{'='*60}\")\n\nprint(f\"\\nClassification Report:\")\nprint(classification_report(\n    all_labels, all_preds,\n    target_names=[\"Negative\", \"Neutral\", \"Positive\"]\n))\n\n# ADEF routing statistics + K_tv quantiles (tau calibration guide)\nroute_a_total = np.mean(all_route_a) * 100\nquantiles = np.percentile(all_conflicts, [50, 80, 90, 95, 99])\nprint(f\"\\n{'='*60}\")\nprint(f\"ADEF ROUTING STATISTICS (tau = {CFG.TAU})\")\nprint(f\"{'='*60}\")\nprint(f\"Route A (Normal DS Fusion):     {route_a_total:.1f}% of samples\")\nprint(f\"Route B (Conflict-Aware):       {100 - route_a_total:.1f}% of samples\")\nprint(f\"Mean conflict K_tv:             {all_conflicts.mean():.4f} +/- {all_conflicts.std():.4f}\")\nprint(f\"K_tv quantiles p50/p80/p90/p95/p99: {np.round(quantiles, 4).tolist()}\")\nprint(f\"  -> tau calibration guide: set CFG.TAU near p80 = {quantiles[1]:.4f}\")\n\n# Uncertainty analysis\ncorrect_mask = all_preds == all_labels\nprint(f\"\\n{'='*60}\")\nprint(f\"UNCERTAINTY ANALYSIS\")\nprint(f\"{'='*60}\")\nif correct_mask.sum() > 0:\n    print(f\"Mean uncertainty (correct):   {all_uncertainties[correct_mask].mean():.4f} +/- {all_uncertainties[correct_mask].std():.4f}\")\nif (~correct_mask).sum() > 0:\n    print(f\"Mean uncertainty (incorrect): {all_uncertainties[~correct_mask].mean():.4f} +/- {all_uncertainties[~correct_mask].std():.4f}\")\nprint(f\"{'='*60}\")'''\n\n# ============================================================\n# CELL 14 - VISUALIZATION\n# ============================================================\ncells_src[14] = r'''# ============================================================\n# VISUALIZATION\n# ============================================================\n\nfig, axes = plt.subplots(2, 4, figsize=(24, 10))\nfig.suptitle(\"ADEF Co-Attention v2 + EDL - Training & Evaluation Results\", fontsize=16, fontweight=\"bold\")\n\n# 1. Loss Curve\naxes[0, 0].plot(history[\"train_loss\"], label=\"Train\", marker=\"o\", markersize=3)\naxes[0, 0].plot(history[\"val_loss\"], label=\"Val\", marker=\"s\", markersize=3)\naxes[0, 0].set_xlabel(\"Epoch\"); axes[0, 0].set_ylabel(\"Loss\")\naxes[0, 0].set_title(\"Loss Curve\"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)\n\n# 2. Accuracy Curve\naxes[0, 1].plot(history[\"train_acc\"], label=\"Train\", marker=\"o\", markersize=3)\naxes[0, 1].plot(history[\"val_acc\"], label=\"Val\", marker=\"s\", markersize=3)\naxes[0, 1].set_xlabel(\"Epoch\"); axes[0, 1].set_ylabel(\"Accuracy\")\naxes[0, 1].set_title(\"Accuracy Curve\"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)\n\n# 3. F1 Curve (Macro = selection metric, Weighted for reference)\naxes[0, 2].plot(history[\"train_f1_macro\"], label=\"Train Macro-F1\", marker=\"o\", markersize=3)\naxes[0, 2].plot(history[\"val_f1_macro\"], label=\"Val Macro-F1\", marker=\"s\", markersize=3)\naxes[0, 2].plot(history[\"val_f1\"], label=\"Val Weighted-F1\", linestyle=\"--\", alpha=0.6)\naxes[0, 2].set_xlabel(\"Epoch\"); axes[0, 2].set_ylabel(\"F1 Score\")\naxes[0, 2].set_title(\"F1 Score Curve (Macro = selection metric)\"); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)\n\n# 4. Conflict K_tv Curve (threshold line from CFG.TAU)\naxes[0, 3].plot(history[\"train_conflict\"], label=\"Train K_tv\", marker=\"o\", markersize=3, color=\"orange\")\naxes[0, 3].plot(history[\"val_conflict\"], label=\"Val K_tv\", marker=\"s\", markersize=3, color=\"red\")\naxes[0, 3].axhline(y=CFG.TAU, color=\"gray\", linestyle=\"--\", alpha=0.5, label=f\"Threshold (tau={CFG.TAU})\")\naxes[0, 3].set_xlabel(\"Epoch\"); axes[0, 3].set_ylabel(\"K_tv\")\naxes[0, 3].set_title(\"Conflict Level (K_tv)\"); axes[0, 3].legend(); axes[0, 3].grid(True, alpha=0.3)\n\n# 5. Confusion Matrix\ncm = confusion_matrix(all_labels, all_preds)\nsns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\",\n            xticklabels=[\"Negative\", \"Neutral\", \"Positive\"],\n            yticklabels=[\"Negative\", \"Neutral\", \"Positive\"],\n            ax=axes[1, 0])\naxes[1, 0].set_xlabel(\"Predicted\"); axes[1, 0].set_ylabel(\"Actual\")\naxes[1, 0].set_title(\"Confusion Matrix\")\n\n# 6. Uncertainty Distribution (Correct vs Incorrect)\ncorrect_mask = all_preds == all_labels\naxes[1, 1].hist(all_uncertainties[correct_mask], bins=30, alpha=0.6,\n                label=f\"Correct (n={correct_mask.sum()})\", color=\"green\", density=True)\nif (~correct_mask).sum() > 0:\n    axes[1, 1].hist(all_uncertainties[~correct_mask], bins=30, alpha=0.6,\n                    label=f\"Incorrect (n={(~correct_mask).sum()})\", color=\"red\", density=True)\naxes[1, 1].set_xlabel(\"Uncertainty (u)\"); axes[1, 1].set_ylabel(\"Density\")\naxes[1, 1].set_title(\"Uncertainty Distribution\"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)\n\n# 7. Per-Class F1 Scores (neutral must be > 0)\nclass_names = [\"Negative\", \"Neutral\", \"Positive\"]\nbars = axes[1, 2].bar(class_names, f1_per_class, color=[\"#e74c3c\", \"#3498db\", \"#2ecc71\"])\naxes[1, 2].set_xlabel(\"Class\"); axes[1, 2].set_ylabel(\"F1 Score\")\naxes[1, 2].set_title(\"Per-Class F1 Score\"); axes[1, 2].set_ylim(0, 1)\nfor bar, val in zip(bars, f1_per_class):\n    axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,\n                    f\"{val:.3f}\", ha=\"center\", va=\"bottom\", fontweight=\"bold\")\naxes[1, 2].grid(True, alpha=0.3, axis=\"y\")\n\n# 8. Conflict Distribution & Route Breakdown (split at CFG.TAU)\naxes[1, 3].hist(all_conflicts[all_conflicts <= CFG.TAU], bins=20, alpha=0.6,\n                label=f\"Route A (n={(all_conflicts <= CFG.TAU).sum()})\", color=\"blue\", density=True)\naxes[1, 3].hist(all_conflicts[all_conflicts > CFG.TAU], bins=20, alpha=0.6,\n                label=f\"Route B (n={(all_conflicts > CFG.TAU).sum()})\", color=\"orange\", density=True)\naxes[1, 3].axvline(x=CFG.TAU, color=\"red\", linestyle=\"--\", alpha=0.7, label=f\"tau={CFG.TAU}\")\naxes[1, 3].set_xlabel(\"Conflict (K_tv)\"); axes[1, 3].set_ylabel(\"Density\")\naxes[1, 3].set_title(\"ADEF Routing Distribution\"); axes[1, 3].legend(); axes[1, 3].grid(True, alpha=0.3)\n\nplt.tight_layout()\nplt.savefig(\"adef_co_attention_results.png\", dpi=150, bbox_inches=\"tight\")\nplt.show()\n\nprint(\"\\u2705 Visualization complete.\")'''\n\n# ============================================================\n# APPLY PATCH\n# ============================================================\nwith io.open(NB, \"r\", encoding=\"utf-8\") as f:\n    nb = json.load(f)\n\nassert len(nb[\"cells\"]) == 15, f\"expected 15 cells, got {len(nb['cells'])}\"\n\nfor idx, src in cells_src.items():\n    cell = nb[\"cells\"][idx]\n    assert cell[\"cell_type\"] == \"code\", f\"cell {idx} is not code\"\n    cell[\"source\"] = src.splitlines(keepends=True)\n    cell[\"outputs\"] = []\n    cell[\"execution_count\"] = None\n    print(f\"patched cell {idx}: {len(src)} chars\")\n\n# also clear outputs of untouched code cells for a clean slate\nfor i, cell in enumerate(nb[\"cells\"]):\n    if cell[\"cell_type\"] == \"code\" and i not in cells_src:\n        cell[\"outputs\"] = []\n        cell[\"execution_count\"] = None\n\nwith io.open(NB, \"w\", encoding=\"utf-8\") as f:\n    json.dump(nb, f, indent=1, ensure_ascii=False)\n\nprint(\"NOTEBOOK PATCHED OK\")\n",
  "filePath": "C:\\Users\\erlanggadewasakti\\AppData\\Local\\Temp\\opencode\\patch_adef_nb.py"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · Kimi K3 (2x usage) · 11.5s)

_Thinking:_

Wait — before running the patch, I should back up the original notebook first (it's the user's file, and the patch is destructive). Let me create a backup copy, then run the patch.

Before patching, back up the original notebook, then run the patch:

**Tool: bash**

**Input:**
```json
{
  "command": "Copy-Item \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\" \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention_v1_backup.ipynb\" -Force; & \"C:\\Users\\erlanggadewasakti\\anaconda3\\python.exe\" \"C:\\Users\\erlanggadewasakti\\AppData\\Local\\Temp\\opencode\\patch_adef_nb.py\""
}
```

**Output:**
```
patched cell 2: 2732 chars
patched cell 3: 2377 chars
patched cell 5: 2912 chars
patched cell 6: 2488 chars
patched cell 8: 3426 chars
patched cell 9: 3343 chars
patched cell 10: 4413 chars
patched cell 12: 6629 chars
patched cell 13: 3692 chars
patched cell 14: 4422 chars
NOTEBOOK PATCHED OK

```

---

## Assistant (Build · Kimi K3 (2x usage) · 34.5s)

_Thinking:_

Patched. Now validate:
1. JSON validity + nbformat structure
2. Every code cell compiles (ast.parse / compile)
3. Cross-cell consistency checks: e.g., cell 9 unpacks 5 values from adef (cell 8 returns 5), cell 12 uses functions defined in cells 10, cell 13 uses all_route_a, etc.

Let me write a validation script.

Patched. Now static validation — JSON structure + compile every cell + cross-cell API consistency:

**Tool: write**

**Input:**
```json
{
  "content": "# Static validation of the patched notebook:\n# 1. JSON + nbformat structure\n# 2. Every code cell compiles (ast)\n# 3. Cross-cell API consistency checks\nimport json\nimport io\nimport ast\n\nNB = r\"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb\"\n\nwith io.open(NB, \"r\", encoding=\"utf-8\") as f:\n    nb = json.load(f)\n\nprint(\"nbformat:\", nb.get(\"nbformat\"), nb.get(\"nbformat_minor\"))\nprint(\"cells:\", len(nb[\"cells\"]))\n\nsrcs = {}\nok = True\nfor i, cell in enumerate(nb[\"cells\"]):\n    if cell[\"cell_type\"] != \"code\":\n        continue\n    src = \"\".join(cell[\"source\"])\n    srcs[i] = src\n    try:\n        compile(src, f\"<cell {i}>\", \"exec\")\n        print(f\"cell {i:2d}: COMPILE OK ({len(src)} chars)\")\n    except SyntaxError as e:\n        ok = False\n        print(f\"cell {i:2d}: SYNTAX ERROR: {e}\")\n\n# --- cross-cell consistency checks ---\nchecks = [\n    # (description, needle, cell)\n    (\"CFG defines TAU\",            \"TAU = 0.1\",                 2),\n    (\"CFG defines GAMMA\",          \"GAMMA = 1.0\",               2),\n    (\"CFG defines LAMBDA_FUSED\",   \"LAMBDA_FUSED = 1.0\",        2),\n    (\"CFG defines U_MIN\",          \"U_MIN = 0.05\",              2),\n    (\"CFG defines GRAD_CLIP\",      \"GRAD_CLIP = 1.0\",           2),\n    (\"CFG defines SCHED_TMAX\",     \"SCHED_TMAX = 30\",           2),\n    (\"CFG defines SELECT_METRIC\",  'SELECT_METRIC = \"macro_f1\"',2),\n    (\"CFG defines UCE_BINS\",       \"UCE_BINS = 10\",             2),\n    (\"CFG defines FILTER flag\",    \"FILTER_CONFLICT_PAIRS = True\", 2),\n    (\"CFG defines CLASS_WEIGHTS\",  \"USE_CLASS_WEIGHTS = True\",  2),\n    (\"filter uses CFG flag\",       \"if CFG.FILTER_CONFLICT_PAIRS:\", 3),\n    (\"ADEF returns 5 values\",      \"return p_final, b_fusion, u_fusion, K_tv, route_a_mask\", 8),\n    (\"model unpacks 5 values\",     \"p_final, b_fusion, u_fusion, K_tv, route_a_mask = self.adef(\", 9),\n    (\"model passes tau=CFG.TAU\",   \"tau=CFG.TAU\",               9),\n    (\"loss has opinion_to_dirichlet\", \"def opinion_to_dirichlet(\",  10),\n    (\"loss applies sample_weight\", \"loss = loss * sample_weight\",   10),\n    (\"training uses L_fused\",      \"L_fused = criterion(alpha_f, labels, epoch)\", 12),\n    (\"training uses CFG.GAMMA\",    \"CFG.GAMMA * L_con\",         12),\n    (\"training uses CFG.LAMBDA_FUSED\", \"CFG.LAMBDA_FUSED * L_fused\", 12),\n    (\"training clip from CFG\",     \"max_norm=CFG.GRAD_CLIP\",    12),\n    (\"training selects macro\",     'CFG.SELECT_METRIC == \"macro_f1\"', 12),\n    (\"eval uses route mask\",       'route_a = out[\"route_a_mask\"].squeeze(1)', 13),\n    (\"eval quantile report\",       \"np.percentile(all_conflicts, [50, 80, 90, 95, 99])\", 13),\n    (\"eval neutral f1 print\",      \"F1 (Neutral):\",             13),\n    (\"viz uses CFG.TAU\",           \"axes[1, 3].axvline(x=CFG.TAU\", 14),\n    (\"viz macro curve\",            'history[\"val_f1_macro\"]',   14),\n    (\"viz saves png\",              'plt.savefig(\"adef_co_attention_results.png\"', 14),\n    (\"encoders sequence H_t\",      \"H_t = self.projection(outputs.last_hidden_state)\", 5),\n    (\"encoders return triple\",     \"return H_t, h_t, attention_mask\", 5),\n    (\"coattn masked softmax\",      \"masked_fill(pad, -1e9)\",    6),\n    (\"coattn thesis affinity\",     \"torch.bmm(self.W_attn(H_t), H_v.transpose(1, 2)) / (self.d_proj ** 0.5)\", 6),\n]\nall_ok = True\nfor desc, needle, idx in checks:\n    found = needle in srcs.get(idx, \"\")\n    all_ok = all_ok and found\n    print((\"PASS \" if found else \"FAIL \") + f\"[cell {idx:2d}] {desc}\")\n\n# no leftover hardcoded tau=0.5 in patched cells\nfor idx in (8, 9, 13, 14):\n    if \"0.5\" in srcs.get(idx, \"\") and \"tau=0.5\" in srcs.get(idx, \"\"):\n        print(f\"WARN: possible hardcoded tau=0.5 in cell {idx}\")\n        all_ok = False\n\nprint(\"\\nRESULT:\", \"ALL CHECKS PASSED\" if (ok and all_ok) else \"SOME CHECKS FAILED\")\n",
  "filePath": "C:\\Users\\erlanggadewasakti\\AppData\\Local\\Temp\\opencode\\validate_adef_nb.py"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · Kimi K3 (2x usage) · 8.0s)

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\python.exe\" \"C:\\Users\\erlanggadewasakti\\AppData\\Local\\Temp\\opencode\\validate_adef_nb.py\""
}
```

**Output:**
```
nbformat: 4 5
cells: 15
cell  1: COMPILE OK (1254 chars)
cell  2: COMPILE OK (2732 chars)
cell  3: COMPILE OK (2377 chars)
cell  4: COMPILE OK (3428 chars)
cell  5: COMPILE OK (2912 chars)
cell  6: COMPILE OK (2488 chars)
cell  7: COMPILE OK (1314 chars)
cell  8: COMPILE OK (3426 chars)
cell  9: COMPILE OK (3343 chars)
cell 10: COMPILE OK (4413 chars)
cell 11: COMPILE OK (1317 chars)
cell 12: COMPILE OK (6629 chars)
cell 13: COMPILE OK (3692 chars)
cell 14: COMPILE OK (4422 chars)
PASS [cell  2] CFG defines TAU
PASS [cell  2] CFG defines GAMMA
PASS [cell  2] CFG defines LAMBDA_FUSED
PASS [cell  2] CFG defines U_MIN
PASS [cell  2] CFG defines GRAD_CLIP
PASS [cell  2] CFG defines SCHED_TMAX
PASS [cell  2] CFG defines SELECT_METRIC
PASS [cell  2] CFG defines UCE_BINS
PASS [cell  2] CFG defines FILTER flag
PASS [cell  2] CFG defines CLASS_WEIGHTS
PASS [cell  3] filter uses CFG flag
PASS [cell  8] ADEF returns 5 values
PASS [cell  9] model unpacks 5 values
PASS [cell  9] model passes tau=CFG.TAU
PASS [cell 10] loss has opinion_to_dirichlet
PASS [cell 10] loss applies sample_weight
PASS [cell 12] training uses L_fused
PASS [cell 12] training uses CFG.GAMMA
PASS [cell 12] training uses CFG.LAMBDA_FUSED
PASS [cell 12] training clip from CFG
PASS [cell 12] training selects macro
PASS [cell 13] eval uses route mask
PASS [cell 13] eval quantile report
PASS [cell 13] eval neutral f1 print
PASS [cell 14] viz uses CFG.TAU
PASS [cell 14] viz macro curve
PASS [cell 14] viz saves png
PASS [cell  5] encoders sequence H_t
PASS [cell  5] encoders return triple
PASS [cell  6] coattn masked softmax
PASS [cell  6] coattn thesis affinity
WARN: possible hardcoded tau=0.5 in cell 8

RESULT: SOME CHECKS FAILED

```

---

## Assistant (Build · Kimi K3 (2x usage) · 13.7s)

_Thinking:_

The "WARN: possible hardcoded tau=0.5 in cell 8" is a false positive from my own check logic — my check triggers if "0.5" AND "tau=0.5" both appear. Let me verify: cell 8 contains "tau=0.5"? Looking at my cell 8 source: the comment says "v1 hardcoded tau=0.5 with mean K_tv ~ 0.014 -> Route B dead." — yes, it's in a COMMENT explaining the v1 problem. That's intentional documentation, not actual code. The default in `__init__` is `tau=0.1`. Let me verify the actual init signature to be sure.

The WARN is from a comment in cell 8 documenting the v1 problem ("v1 hardcoded tau=0.5..."). Let me verify it's only in the comment, not in code:

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\python.exe\" -c \"import json,io; nb=json.load(io.open(r'D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb',encoding='utf-8')); src=''.join(nb['cells'][8]['source']); [print(repr(l)) for l in src.splitlines() if '0.5' in l]\""
}
```

**Output:**
```
'# K_tv scale. v1 hardcoded tau=0.5 with mean K_tv ~ 0.014 -> Route B dead.'

```

---

## Assistant (Build · Kimi K3 (2x usage) · 15.6s)

_Thinking:_

Confirmed — the only "tau=0.5" occurrence is in the documentation comment explaining the v1 bug. The actual default is `tau=0.1` and the model is instantiated with `tau=CFG.TAU`. All checks effectively pass.

Now let me also do a deeper functional smoke check that doesn't require torch — I can't run the model without torch. But I can do one more meaningful static check: verify tensor-shape logic of the co-attention and ADEF modules by reasoning (done during design) — plus verify the diff between old and new notebook to make sure nothing else changed accidentally. Let me do a quick diff summary: compare cell sources between backup and patched for cells 0,1,4,7,11 (should be identical).

**Tool: bash**

**Input:**
```json
{
  "command": "& \"C:\\Users\\erlanggadewasakti\\anaconda3\\python.exe\" -c \"\nimport json, io\nold = json.load(io.open(r'D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention_v1_backup.ipynb', encoding='utf-8'))\nnew = json.load(io.open(r'D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\\adef_co_attention.ipynb', encoding='utf-8'))\nfor i in range(15):\n    so = ''.join(old['cells'][i]['source']); sn = ''.join(new['cells'][i]['source'])\n    status = 'IDENTICAL' if so == sn else 'MODIFIED '\n    print(f'cell {i:2d}: {status} ({len(so)} -> {len(sn)} chars)')\nprint('kernelspec:', new['metadata'].get('kernelspec', {}).get('name'))\n\""
}
```

**Output:**
```
cell  0: IDENTICAL (59 -> 59 chars)
cell  1: IDENTICAL (1254 -> 1254 chars)
cell  2: MODIFIED  (1078 -> 2732 chars)
cell  3: MODIFIED  (2331 -> 2377 chars)
cell  4: IDENTICAL (3428 -> 3428 chars)
cell  5: MODIFIED  (2131 -> 2912 chars)
cell  6: MODIFIED  (2001 -> 2488 chars)
cell  7: IDENTICAL (1314 -> 1314 chars)
cell  8: MODIFIED  (3395 -> 3426 chars)
cell  9: MODIFIED  (3104 -> 3343 chars)
cell 10: MODIFIED  (3246 -> 4413 chars)
cell 11: IDENTICAL (1317 -> 1317 chars)
cell 12: MODIFIED  (5198 -> 6629 chars)
cell 13: MODIFIED  (3106 -> 3692 chars)
cell 14: MODIFIED  (4098 -> 4422 chars)
kernelspec: python3

```

---

## Assistant (Build · Kimi K3 (2x usage) · 35.6s)

_Thinking:_

The diff is exactly as planned: cells 0,1,4,7,11 identical; cells 2,3,5,6,8,9,10,12,13,14 modified. Notebook metadata intact.

One more subtle thing to double-check: cell 12 references `train_df` which is defined in cell 4 — yes, `train_df` exists there. Good.

Also `opinion_to_dirichlet` is defined in cell 10, used in cell 12 — good. `compute_uce` in cell 11 takes `num_bins` param — cell 13 passes `num_bins=CFG.UCE_BINS` — matches cell 11 signature (`def compute_uce(predictions, labels, uncertainties, num_bins=10)`) — good.

Cell 13 uses `all_route_a` array — collected from out["route_a_mask"] — good. Cell 14 uses `f1_per_class`, `all_conflicts`, `all_uncertainties`, `all_labels`, `all_preds`, `history` — all defined in cells 12/13 — good.

One potential runtime issue in cell 14: `f1_per_class` computed in cell 13 — yes. `correct_mask` recomputed in cell 14 — yes.

Another check: in cell 12, `criterion(alpha_f, labels, epoch)` — EvidentialLoss.forward(alpha, labels, epoch) — matches.

In cell 10, EvidentialLoss.forward uses self.class_weights[labels] — class_weights is on device, labels on device — fine.

In cell 8, ADEFModule.forward returns 5 values — cell 9 unpacks 5 — good.

In cell 6, `F.softmax` — F imported cell 1 — good.

In cell 5, TextEncoder returns 3 values — cell 9 `H_t, h_t, text_mask = self.text_encoder(...)` — good. ImageEncoder returns 2 — `H_v, h_v = self.image_encoder(image)` — good.

Tensor shape trace for co-attention: H_t [B,150,512], W_attn(H_t) [B,150,512], H_v.transpose(1,2) [B,512,49] → bmm → S [B,150,49]. A_tv [B,150,49] @ H_v [B,49,512] → [B,150,512]. mask [B,150,1] broadcast multiply → sum dim=1 → [B,512]. S_T = S.transpose(1,2) [B,49,150]; pad = (text_mask==0).unsqueeze(1) [B,1,150] → masked_fill broadcast over dim 1 — [B,49,150] masked_fill with [B,1,150] — broadcasting works in masked_fill. A_vt [B,49,150] @ H_t [B,150,512] → [B,49,512] → mean dim=1 → [B,512]. cat → [B,1024] → fusion_proj → [B,512]. ✓

Loss numeric check: alpha_f = b_f * (3/clamp(u_f,0.05,1)) + 1. Route A: u_fusion after two DS combines could be tiny → clamped to 0.05 → S=60 → alpha ≤ ~61. lgamma(61) fine. ✓

One consideration: in training, `model.train()` puts RoBERTa submodule in train mode → dropout active inside frozen RoBERTa (under no_grad). This was also true in v1 and all baselines — consistent, deterministic enough given seed. Not changing (keeps comparability).

Also `weights` variable in cell 12 shadows torchvision `weights`? No conflict — no variable named `weights` elsewhere in that scope. Fine.

Also one more check — `counts = train_df["label"].value_counts().sort_index()` — labels 0,1,2 all present in train — yes (950/329/1878).

The md plan references `U_MIN` — I included it in CFG ✓.

Now the deliverables are complete:
1. ✅ md rewritten
2. ✅ notebook patched + validated
3. Backup of v1 created

I should also clean up the temp scripts? They're in the approved temp dir; can leave, but cleaner to remove. I'll leave them — actually they're harmless in temp. Remove them to be tidy? The temp dir is pre-approved scratch space; leaving two small .py files is fine, but tidy is better. I'll remove them.

Wait — one more thing to double check: notebook cell 3 — I removed the `failed_samples = []` list and the empty-IDs print block (cosmetic cleanup, behavior preserved). That's fine, noted as intentional.

Also should I update AGENTS.md? There's no AGENTS.md in this project (not in file listing). Skip.

Final summary for the user, in a concise structured form. Mark todos complete.

Diff is exactly as designed — only the 10 intended cells changed. Let me clean up the temp scripts and finalize:

**Tool: bash**

**Input:**
```json
{
  "command": "Remove-Item \"C:\\Users\\erlanggadewasakti\\AppData\\Local\\Temp\\opencode\\patch_adef_nb.py\", \"C:\\Users\\erlanggadewasakti\\AppData\\Local\\Temp\\opencode\\validate_adef_nb.py\" -Force; Get-ChildItem \"D:\\Coding\\Project\\All Thesis\\Comparative Analysis of Cross-Modal Fusion\" | ForEach-Object { \"$($_.Name) - $([math]::Round($_.Length/1KB,1)) KB\" }"
}
```

**Output:**
```
.gitignore - 0 KB
adef_co_attention_results.png - 217.4 KB
adef_co_attention_v1_backup.ipynb - 335.2 KB
adef_co_attention.ipynb - 56.7 KB
co_attention.ipynb - 249.9 KB
cross_attention.ipynb - 250.7 KB
early_fusion.ipynb - 257.8 KB
late_fusion.ipynb - 256.3 KB
prompt.md - 7.6 KB
rancangan_tesis_adef_co_attention.md - 18 KB

```

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Rewrite rancangan_tesis_adef_co_attention.md (v2 plan, in place)",
      "status": "completed",
      "priority": "high"
    },
    {
      "content": "Patch adef_co_attention.ipynb (CFG all-hyperparams, seq-level encoders, true co-attention, fixed losses/training/eval/viz)",
      "status": "completed",
      "priority": "high"
    },
    {
      "content": "Static validation: notebook JSON valid + every patched cell compiles (ast)",
      "status": "completed",
      "priority": "high"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Rewrite rancangan_tesis_adef_co_attention.md (v2 plan, in place)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Patch adef_co_attention.ipynb (CFG all-hyperparams, seq-level encoders, true co-attention, fixed losses/training/eval/viz)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Static validation: notebook JSON valid + every patched cell compiles (ast)",
    "status": "completed",
    "priority": "high"
  }
]
```

---

## Assistant (Build · Kimi K3 (2x usage) · 26.8s)

_Thinking:_

All done. Let me write the final summary. The user asked in English (mixed), so I'll respond in English. Summary should cover:

1. Why neutral was mispredicted (diagnosis summary)
2. What was changed in the notebook
3. What was changed in the plan md
4. Files: backup created, outputs cleared (ready to re-run)
5. What to do next: run on the RTX 3060 machine, expected outcomes, tau calibration guide

Keep it concise but complete.

Done. Both deliverables are complete and statically validated (all 15 cells compile, all 31 consistency checks pass).

## Files changed

| File | Change |
|---|---|
| `adef_co_attention.ipynb` | Patched in place (10 of 15 cells); outputs cleared, ready to re-run |
| `adef_co_attention_v1_backup.ipynb` | Original v1 preserved as backup |
| `rancangan_tesis_adef_co_attention.md` | Rewritten as **ADEF v2** plan (same Indonesian academic style, same structure + Lampiran A revision log) |

## Why your model never predicted Neutral (F1 = 0.00)

1. **59/30/10 class imbalance with zero compensation** — the SoS evidential loss collapsed to the majority class
2. **Best checkpoint selected by weighted-F1** — a majority-biased metric, so a 0%-neutral-recall model "won"
3. **Degenerate co-attention** — ran on pooled `[B,512]` vectors, so the attention "matrix" was a single sigmoid scalar `[B,1,1]` — no token↔patch alignment
4. **LR = 2e-5 on randomly-initialized heads** (backbones frozen) → undertrained
5. **τ = 0.5 hardcoded while K_tv ≈ 0.014** → Route B fired on 0% of samples
6. **p_final never appeared in the loss** → the fused decision was never directly optimized

## Key fixes (notebook = plan doc, 1:1)

- **CFG is now the single source of truth for every hyperparameter**: `TAU=0.1` (calibratable), `GAMMA`, `LAMBDA_FUSED`, `LR=1e-4`, `WEIGHT_DECAY`, `GRAD_CLIP`, `SCHED_TMAX`, `ANNEALING_EPOCHS`, `DROPOUT`, `USE_CLASS_WEIGHTS`, `SELECT_METRIC`, `UCE_BINS`, `U_MIN`, `FILTER_CONFLICT_PAIRS`
- **Sequence-level co-attention** faithful to your thesis formula: `S = H_t·W·H_vᵀ/√d ∈ [B,150,49]`, bidirectional softmax with padding mask — encoders now return token/patch features + pooled vectors
- **Class-weighted evidential loss** (`w ≈ [1.11, 3.20, 0.56]`, neutral weighs ~5.7× positive)
- **Fused-opinion supervision**: `α_fusion = b_f·(M/û)+1` added to the loss, so the final ADEF decision is directly trained
- **Best model selected by Macro-F1**
- **τ calibration tooling**: test eval now prints K_tv quantiles (p50/p80/p90/p95/p99) with the p80 recommendation; all plots read `CFG.TAU` (no hardcoded 0.5)
- Numerical guards: `clamp(1−K)`, `u_fusion ∈ [ε,1]`, masked softmax with −1e9

## Next step (on your RTX 3060 machine)

Just run the notebook top-to-bottom — no installs needed there. Success targets: **Neutral F1 ≈ 0.30–0.40** (on par with your other 4 models), **Macro-F1 ≥ 0.55**, and Route B firing on a non-trivial share of test samples. If Route B still fires on <1%, follow the printed p80 guidance to adjust `CFG.TAU` — that's your τ-sensitivity ablation for the thesis.

---

