#!/usr/bin/env python3
"""Transform v2 notebook -> v3 by replacing specific cells.
Reads v2, replaces key cells with v3 code, writes v3.
All cell sources use triple-single-quoted strings with literal Unicode."""
import nbformat as nbf, os, sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adef_co_attention.ipynb")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adef_co_attention_v3.ipynb")

print("Reading v2...")
nb = nbf.read(SRC, as_version=4)
cells = nb.cells

# ──────────────────────────────────────────────
# Cell 0: markdown title
# ──────────────────────────────────────────────
cells[0].source = '''# ADEF v3

**Adaptive Evidential Fusion with Learned Soft Conflict Gating & Deep Gated Co-Attention**

v2 → v3 improvements:
1. **Soft-learnable conflict gate** — differentiable σ((K−τ)/s), replaces hard τ-mask
2. **Deep gated co-attention** — 2 stacked layers with tanh-gated residual updates
3. **Vacuous-opinion dropout** — train-time modality dropout via DST vacuous beliefs (b=0,u=1)
4. **Jøsang uncertainty discounting** — learned per-branch reliability scalars
5. **Digamma evidential loss** — expected cross-entropy (less overconfidence than SoS)
6. **Label smoothing 0.05 + sqrt-class-weights + GAMMA=0.1**
7. **Early stopping** (patience 6) + **3-seed** evaluation (mean±std)
'''

# ──────────────────────────────────────────────
# Cell 1: Imports (add `copy` for deepcopy)
# ──────────────────────────────────────────────
cells[1].source = '''import torch
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
import copy
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
'''

# ──────────────────────────────────────────────
# Cell 2: CFG v3
# ──────────────────────────────────────────────
cells[2].source = '''# ============================================================
# CONFIGURATION v3 (single source of truth for ALL hyperparameters)
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
    FILTER_CONFLICT_PAIRS = True     # True = baseline-compatible; False = ablation (real conflict)

    # =========================
    # OPTIMIZATION HYPERPARAMETERS
    # =========================
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-3              # v3: stronger regularization (was 1e-4)
    GRAD_CLIP = 1.0
    SCHED_TMAX = 30
    DROPOUT = 0.45                   # v3: higher dropout (was 0.3; 3 heads + co-attn >> baseline params)
    SEED = 42
    SEEDS = [42, 1337, 2024]         # v3: 3-seed evaluation

    # =========================
    # EARLY STOPPING (v3)
    # =========================
    EARLY_STOP_PATIENCE = 6

    # =========================
    # ARCHITECTURE HYPERPARAMETERS
    # =========================
    MAX_LEN = 150
    D_BERT = 768
    D_CNN = 1024
    D_PROJ = 512
    NUM_CLASSES = 3
    COATTN_LAYERS = 2                # v3: stacked gated co-attention

    # =========================
    # EDL / ADEF v3 HYPERPARAMETERS
    # =========================
    ANNEALING_EPOCHS = 10
    EDL_LOSS_TYPE = "digamma"        # v3: "digamma" (expected-CE) | "sos" (Bayes risk) | "both"
    LABEL_SMOOTHING = 0.05           # v3: smooth one-hot labels in loss
    TAU = 0.036                      # init from v2 val p80; learnable gate uses this as center
    GATE_TYPE = "parametric"         # v3: "parametric" (sigma((K-tau)/s)) | "mlp"
    GATE_LEARNABLE = True            # v3: tau & s learnable
    GATE_TEMP_INIT = 0.02            # v3: initial temperature for sigmoid gate
    OPINION_DROP_PROB = 0.15         # v3: vacuous-opinion dropout prob (train only)
    USE_DISCOUNTING = True           # v3: Josang pre-fusion discounting
    GAMMA = 0.1                      # v3: L_con weight (was 1.0 -- suppressed conflict!)
    LAMBDA_FUSED = 1.0
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHT_MODE = "sqrt_inv"   # v3: "sqrt_inv" (sqrt inverse freq) | "inv"
    U_MIN = 0.05

    # =========================
    # MODEL SELECTION / EVALUATION
    # =========================
    SELECT_METRIC = "macro_f1"
    UCE_BINS = 10

    # =========================
    # PRETRAINED MODELS
    # =========================
    TEXT_MODEL = "roberta-base"
    IMAGE_MODEL = "densenet121"

print(f"\\u2705 CFG v3 loaded. Device: {CFG.DEVICE}  |  Gate: {CFG.GATE_TYPE}  |  Loss: {CFG.EDL_LOSS_TYPE}")
print(f"   CoAttn layers={CFG.COATTN_LAYERS}  Drop={CFG.OPINION_DROP_PROB}  Discount={CFG.USE_DISCOUNTING}  Gamma={CFG.GAMMA}")
'''

# ──────────────────────────────────────────────
# Cells 3, 4, 5: Data loading, Dataset, Encoders — UNCHANGED from v2
# (We keep the original source for these cells)
# ──────────────────────────────────────────────
print("  Cells 0-5: keeping 3-5 unchanged, 0-2 replaced")

# ──────────────────────────────────────────────
# Cell 6: Deep Gated Bi-CoAttention (replace v2 BiCoAttention)
# ──────────────────────────────────────────────
cells[6].source = '''# ============================================================
# DEEP GATED BI-COATTENTION (v3)
# ============================================================
# Stacks N layers of simultaneous gated bidirectional co-attention.
# Each layer applies text->visual AND visual->text attention in parallel
# using the SAME input features, then updates both modalities via
# tanh-gated residual connections.
#
# Layer l:
#   S   = H_t @ W_l @ H_v^T / sqrt(d)           -- affinity [B, L_t, N_v]
#   v2t = softmax(S, dim=patches) @ H_v          -- text-attended visual features
#   t2v = softmax(S^T, dim=tokens, masked) @ H_t -- visual-attended text features
#   H_t <- H_t + tanh(Linear_t(H_t)) * v2t       -- gated residual (text)
#   H_v <- H_v + tanh(Linear_v(H_v)) * t2v       -- gated residual (visual)
#
# Final h_c: concat(masked-mean(H_t_final), mean(H_v_final)) -> MLP -> [d]
# The gating allows the model to learn HOW MUCH cross-modal information
# to absorb at each layer, preventing co-adaptation overfitting.


class GatedBiCoAttention(nn.Module):
    def __init__(self, d_proj=512, dropout=0.45, num_layers=2):
        super().__init__()
        self.d_proj = d_proj
        self.num_layers = num_layers

        for layer_idx in range(num_layers):
            setattr(self, f"W_attn_{layer_idx}", nn.Linear(d_proj, d_proj, bias=False))
            setattr(self, f"gate_t_{layer_idx}", nn.Linear(d_proj, d_proj))
            setattr(self, f"gate_v_{layer_idx}", nn.Linear(d_proj, d_proj))

        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * d_proj, d_proj),
            nn.ReLU(),
            nn.LayerNorm(d_proj),
            nn.Dropout(dropout)
        )

    def forward(self, H_t, H_v, text_mask):
        # H_t: [B, L_t, d_proj]   H_v: [B, N_v, d_proj]   text_mask: [B, L_t]
        pad_mask = (text_mask == 0).unsqueeze(1)  # [B, 1, L_t]

        for layer_idx in range(self.num_layers):
            W = getattr(self, f"W_attn_{layer_idx}")
            gate_t = getattr(self, f"gate_t_{layer_idx}")
            gate_v = getattr(self, f"gate_v_{layer_idx}")

            # Affinity matrix
            S = torch.bmm(W(H_t), H_v.transpose(1, 2)) / (self.d_proj ** 0.5)  # [B, L_t, N_v]

            # Text -> Visual (each token attends over all patches)
            A_tv = F.softmax(S, dim=-1)                # [B, L_t, N_v]
            v2t = torch.bmm(A_tv, H_v)                  # [B, L_t, d]

            # Visual -> Text (each patch attends over non-pad tokens)
            S_T = S.transpose(1, 2).masked_fill(pad_mask, -1e9)  # [B, N_v, L_t]
            A_vt = F.softmax(S_T, dim=-1)              # [B, N_v, L_t]
            t2v = torch.bmm(A_vt, H_t)                  # [B, N_v, d]

            # Gated residual (SIMULTANEOUS: both use input H_t, H_v, not updated mid-layer)
            H_t = H_t + torch.tanh(gate_t(H_t)) * v2t
            H_v = H_v + torch.tanh(gate_v(H_v)) * t2v

        # Pooling: masked-mean for text, mean for visual
        mask = text_mask.unsqueeze(-1).float()
        t_summary = (H_t * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-10)
        v_summary = H_v.mean(dim=1)
        h_c = self.fusion_proj(torch.cat([t_summary, v_summary], dim=1))
        return h_c


print("\\u2705 GatedBiCoAttention (v3, {CFG.COATTN_LAYERS}-layer) defined.".format(CFG=CFG))
'''

# ──────────────────────────────────────────────
# Cell 7: ENNHead + SL + ReliabilityDiscounter (v3)
# ──────────────────────────────────────────────
cells[7].source = '''# ============================================================
# ENN HEAD & SUBJECTIVE LOGIC UTILITIES
# ============================================================

class ENNHead(nn.Module):
    def __init__(self, d_proj=512, num_classes=3, dropout=0.45):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_proj, d_proj // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj // 2, num_classes),
            nn.Softplus()
        )

    def forward(self, h):
        evidence = self.fc(h)
        alpha = evidence + 1
        return alpha


def compute_belief_uncertainty(alpha, num_classes=3, eps=1e-8):
    S = torch.sum(alpha, dim=1, keepdim=True)
    e = alpha - 1
    b = e / (S + eps)
    u = num_classes / (S + eps)
    return b, u, S


# ============================================================
# JOSANG'S DISCOUNTING OPERATOR (v3)
# ============================================================
# Applies pre-fusion reliability discounting per branch.
#   b' = r * b
#   u' = 1 - r * (1 - u)  =  r*u + (1-r)
# r = 1: identity (fully trust the branch)
# r -> 0: vacuous opinion (b=0, u=1) -- "I don't know"
# Learned scalar r = sigmoid(raw_r), initialized to ~0.9


class ReliabilityDiscounter(nn.Module):
    def __init__(self, name="t"):
        super().__init__()
        self.name = name
        self.raw_r = nn.Parameter(torch.tensor(2.197, dtype=torch.float32))  # sigmoid(2.197) ~ 0.9

    def forward(self, b, u):
        r = torch.sigmoid(self.raw_r)
        b_discounted = r * b
        u_discounted = r * u + (1.0 - r)
        return b_discounted, u_discounted


print("\\u2705 ENNHead, Subjective Logic utils & ReliabilityDiscounter defined.")
'''

# ──────────────────────────────────────────────
# Cell 8: SoftConflictGate + ADEFModuleV3 (replace v2 ADEFModule)
# ──────────────────────────────────────────────
cells[8].source = '''# ============================================================
# SOFT CONFLICT GATE (v3 -- differentiable, learnable)
# ============================================================
# Replaces the hard route_a_mask with a smooth sigmoid gate:
#   parametric:  g = sigma((K_tv - tau) / s),  tau in [0.01, 0.5], s > 0
#   mlp:         g = MLP([K_tv, u_t, u_v, u_c, L1(p_t, p_v)]) -> [0,1]
#
# Final fusion = (1 - g) * DS_fused  +  g * conflict_resolved
# Both routes ALWAYS contribute via soft blending; gradients flow through g
# back into the ENN heads via K_tv. This makes "Adaptive" truly LEARNED.


class SoftConflictGate(nn.Module):
    def __init__(self, gate_type="parametric", init_tau=0.036, init_temp=0.02):
        super().__init__()
        self.gate_type = gate_type

        if gate_type == "parametric":
            # tau = 0.01 + 0.49 * sigmoid(theta_tau)  ->  tau in [0.01, 0.5]
            tau_raw_val = np.log(init_tau / (0.5 - init_tau)) if init_tau < 0.5 else 5.0
            self.theta_tau = nn.Parameter(torch.tensor(tau_raw_val, dtype=torch.float32))
            # s = softplus(theta_temp) + 1e-4  ->  s > 0
            temp_raw_val = np.log(init_temp)
            self.theta_temp = nn.Parameter(torch.tensor(temp_raw_val, dtype=torch.float32))

        elif gate_type == "mlp":
            self.gate_net = nn.Sequential(
                nn.Linear(5, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def forward(self, K_tv, u_t, u_v, u_c, p_t, p_v):
        if self.gate_type == "parametric":
            tau = 0.01 + 0.49 * torch.sigmoid(self.theta_tau)
            s = F.softplus(self.theta_temp) + 1e-4
            g = torch.sigmoid((K_tv - tau) / s)
        else:  # mlp
            features = torch.cat([
                K_tv,
                u_t, u_v, u_c,
                torch.abs(p_t - p_v).mean(dim=1, keepdim=True)
            ], dim=1)
            g = self.gate_net(features)
        return g   # [B, 1], g ~ 1 = conflict, g ~ 0 = agree


# ============================================================
# ADEF v3 MODULE (soft routing + discounting + Dempster)
# ============================================================

class ADEFModuleV3(nn.Module):
    def __init__(self, num_classes=3, gate_type="parametric", init_tau=0.036,
                 init_temp=0.02, use_discounting=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_discounting = use_discounting

        self.gate = SoftConflictGate(gate_type=gate_type, init_tau=init_tau,
                                     init_temp=init_temp)

        if use_discounting:
            self.discounter_t = ReliabilityDiscounter(name="t")
            self.discounter_v = ReliabilityDiscounter(name="v")
            self.discounter_c = ReliabilityDiscounter(name="c")

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
        eps = 1e-8
        K = self.num_classes

        # --- 1. Discount opinions before fusion ---
        if self.use_discounting:
            b_t_d, u_t_d = self.discounter_t(b_t, u_t)
            b_v_d, u_v_d = self.discounter_v(b_v, u_v)
            b_c_d, u_c_d = self.discounter_c(b_c, u_c)
        else:
            b_t_d, u_t_d = b_t, u_t
            b_v_d, u_v_d = b_v, u_v
            b_c_d, u_c_d = b_c, u_c

        # --- 2. Class probabilities & conflict K_tv for gate ---
        p_t = b_t_d + u_t_d / K
        p_v = b_v_d + u_v_d / K

        b_t_sum_d = torch.sum(b_t_d, dim=1, keepdim=True)
        b_v_sum_d = torch.sum(b_v_d, dim=1, keepdim=True)
        K_tv = b_t_sum_d * b_v_sum_d - torch.sum(b_t_d * b_v_d, dim=1, keepdim=True)

        # --- 3. Soft conflict gate ---
        g = self.gate(K_tv, u_t_d, u_v_d, u_c_d, p_t, p_v)  # [B, 1]

        # --- 4. Route A: standard 2-stage Dempster fusion ---
        b_tv_a, u_tv_a, _ = self.dempster_combine(b_t_d, u_t_d, b_v_d, u_v_d)
        b_final_a, u_final_a, _ = self.dempster_combine(b_tv_a, u_tv_a, b_c_d, u_c_d)

        # --- 5. Route B: conflict-aware blending ---
        b_avg = (b_t_d + b_v_d) / 2.0
        b_final_b = (1.0 - K_tv) * b_avg + K_tv * b_c_d
        u_final_b = torch.clamp(1.0 - torch.sum(b_final_b, dim=1, keepdim=True), min=eps)

        # --- 6. Soft blend: final = (1-g) * RouteA + g * RouteB ---
        b_fusion = (1.0 - g) * b_final_a + g * b_final_b
        u_fusion = torch.clamp((1.0 - g) * u_final_a + g * u_final_b, min=eps, max=1.0)

        # --- 7. Final decision ---
        p_final = b_fusion + u_fusion / K

        return p_final, b_fusion, u_fusion, K_tv, g


print("\\u2705 ADEFModuleV3 with soft conflict gate defined.")
'''

# ──────────────────────────────────────────────
# Cell 9: ADEFCoAttnNetV3 (replace v2 model)
# ──────────────────────────────────────────────
cells[9].source = '''# ============================================================
# ADEFCoAttnNet v3
# ============================================================
# Changes from v2:
#   - GatedBiCoAttention (deep gated co-attention instead of 1-layer)
#   - Soft-learned conflict gate (differentiable routing)
#   - Vacuous-opinion dropout at train time (modality dropout via DST)
#   - Josang reliability discounting before fusion


class ADEFCoAttnNetV3(nn.Module):
    def __init__(self, d_proj=512, num_classes=3, dropout=0.45,
                 coattn_layers=2, gate_type="parametric", init_tau=0.036,
                 init_temp=0.02, use_discounting=True, opinion_drop_prob=0.0):
        super().__init__()
        # Feature extractors (sequence-level)
        self.text_encoder = TextEncoder(d_bert=CFG.D_BERT, d_proj=d_proj)
        self.image_encoder = ImageEncoder(d_cnn=CFG.D_CNN, d_proj=d_proj)

        # Deep gated co-attention
        self.co_attention = GatedBiCoAttention(
            d_proj=d_proj, dropout=dropout, num_layers=coattn_layers
        )

        # 3 independent ENN heads
        self.enn_text = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.enn_image = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)
        self.enn_coattn = ENNHead(d_proj=d_proj, num_classes=num_classes, dropout=dropout)

        # ADEF v3 fusion module (soft gate + discounting)
        self.adef = ADEFModuleV3(
            num_classes=num_classes, gate_type=gate_type, init_tau=init_tau,
            init_temp=init_temp, use_discounting=use_discounting
        )

        self.num_classes = num_classes
        self.opinion_drop_prob = opinion_drop_prob

    def forward(self, input_ids, attention_mask, image):
        # 1. Feature extraction (sequence-level + pooled)
        H_t, h_t, text_mask = self.text_encoder(input_ids, attention_mask)
        H_v, h_v = self.image_encoder(image)

        # 2. Deep gated co-attention
        h_c = self.co_attention(H_t, H_v, text_mask)

        # 3. ENN Heads -> Dirichlet parameters
        alpha_t = self.enn_text(h_t)
        alpha_v = self.enn_image(h_v)
        alpha_c = self.enn_coattn(h_c)

        # 4. Subjective Logic: belief & uncertainty
        b_t, u_t, _ = compute_belief_uncertainty(alpha_t, self.num_classes)
        b_v, u_v, _ = compute_belief_uncertainty(alpha_v, self.num_classes)
        b_c, u_c, _ = compute_belief_uncertainty(alpha_c, self.num_classes)

        # 5. Vacuous-opinion dropout (train only)
        #    Replace each unimodal opinion with vacuous (b=0, u=1) with prob p.
        #    DST identity: Dempster(vacuous, omega) = omega.
        #    This trains the fusion to handle missing/unreliable modalities
        #    and directly regularizes co-adaptation overfitting.
        if self.training and self.opinion_drop_prob > 0.0:
            B = b_t.shape[0]
            dev = b_t.device
            keep_t = (torch.rand(B, 1, device=dev) > self.opinion_drop_prob).float()
            keep_v = (torch.rand(B, 1, device=dev) > self.opinion_drop_prob).float()
            b_t = b_t * keep_t
            u_t = u_t * keep_t + (1.0 - keep_t)
            b_v = b_v * keep_v
            u_v = u_v * keep_v + (1.0 - keep_v)

        # 6. ADEF fusion (soft gate + discounting)
        p_final, b_fusion, u_fusion, K_tv, gate = self.adef(
            b_t, u_t, b_v, u_v, b_c, u_c
        )

        return {
            "alpha_t": alpha_t, "alpha_v": alpha_v, "alpha_c": alpha_c,
            "b_t": b_t, "u_t": u_t,
            "b_v": b_v, "u_v": u_v,
            "b_c": b_c, "u_c": u_c,
            "p_final": p_final,
            "b_fusion": b_fusion, "u_fusion": u_fusion,
            "K_tv": K_tv,
            "gate": gate   # v3: soft gate value [B, 1]
        }


def create_model_v3(overrides=None):
    """Create v3 model with optional override kwargs (for ablation tests)."""
    kwargs = dict(
        d_proj=CFG.D_PROJ, num_classes=CFG.NUM_CLASSES, dropout=CFG.DROPOUT,
        coattn_layers=CFG.COATTN_LAYERS, gate_type=CFG.GATE_TYPE,
        init_tau=CFG.TAU, init_temp=CFG.GATE_TEMP_INIT,
        use_discounting=CFG.USE_DISCOUNTING,
        opinion_drop_prob=CFG.OPINION_DROP_PROB
    )
    if overrides:
        kwargs.update(overrides)
    return ADEFCoAttnNetV3(**kwargs).to(CFG.DEVICE)


model = create_model_v3()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\u2705 ADEFCoAttnNetV3 created.")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")
'''

# ──────────────────────────────────────────────
# Cell 10: EvidentialLossV3 (replace v2 loss)
# ──────────────────────────────────────────────
cells[10].source = '''# ============================================================
# BLENDED LOSS FUNCTION v3
# L_overall = L_sup + lambda_f * L_fused + gamma * L_con
# ============================================================
# Changes from v2:
#   1. Digamma (expected-CE) loss -- less overconfidence-prone than Sum-of-Squares
#   2. Label smoothing (0.05) in loss targets
#   3. Sqrt-inverse-freq class weights (gentler than full inverse; recovers Neutral precision)
#   4. GAMMA = 0.1 (L_con no longer suppresses cross-modal conflict signal)


class EvidentialLossV3(nn.Module):
    def __init__(self, num_classes=3, annealing_epochs=10, class_weights=None,
                 loss_type="digamma", label_smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

    def _smooth_labels(self, y_onehot):
        if self.label_smoothing > 0:
            return (1.0 - self.label_smoothing) * y_onehot + \\
                   self.label_smoothing / self.num_classes
        return y_onehot

    # ---- Expected Cross-Entropy (Digamma) Loss ----
    # L_ce = sum_j y_j * (psi(S) - psi(alpha_j))
    def digamma_loss(self, alpha, y_onehot, sample_weight=None):
        alpha = torch.clamp(alpha, min=1e-10)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loss = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    # ---- Bayes Risk (Sum-of-Squares) Loss (v2 default) ----
    def bayes_risk_loss(self, alpha, y_onehot, sample_weight=None):
        alpha = torch.clamp(alpha, min=1e-10)
        S = torch.sum(alpha, dim=1, keepdim=True)
        p_hat = alpha / S
        err = torch.sum((y_onehot - p_hat) ** 2, dim=1)
        var = torch.sum(p_hat * (1.0 - p_hat) / (S + 1.0), dim=1)
        loss = err + var
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    # ---- KL Divergence to uniform Dirichlet ----
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

    def forward(self, alpha, labels, epoch):
        y_onehot = self._smooth_labels(
            F.one_hot(labels, num_classes=self.num_classes).float()
        )
        lambda_t = min(1.0, epoch / max(self.annealing_epochs, 1))
        sw = self.class_weights[labels] if self.class_weights is not None else None

        if self.loss_type == "digamma":
            loss_err = self.digamma_loss(alpha, y_onehot, sw)
        elif self.loss_type == "sos":
            loss_err = self.bayes_risk_loss(alpha, y_onehot, sw)
        else:  # "both"
            loss_err = (self.digamma_loss(alpha, y_onehot, sw)
                        + self.bayes_risk_loss(alpha, y_onehot, sw)) / 2.0

        loss_kl = self.kl_divergence_reg(alpha, y_onehot)
        return loss_err + lambda_t * loss_kl


# Reconstruct Dirichlet alpha from fused Subjective-Logic opinion
def opinion_to_dirichlet(b, u, num_classes=3, u_min=0.05):
    u_c = torch.clamp(u, min=u_min, max=1.0)
    S = num_classes / u_c
    return b * S + 1.0


# Semantic Conflict Loss
# L_con = d_PD * d_CC = 0.5*(1-u_t)*(1-u_v) * sum|p_t - p_v|
# GAMMA = 0.1 stops this from suppressing cross-modal disagreement
def semantic_conflict_loss(alpha_t, alpha_v, num_classes=3, eps=1e-8):
    S_t = torch.sum(alpha_t, dim=1, keepdim=True)
    S_v = torch.sum(alpha_v, dim=1, keepdim=True)
    p_t = alpha_t / (S_t + eps)
    p_v = alpha_v / (S_v + eps)
    u_t = num_classes / (S_t + eps)
    u_v = num_classes / (S_v + eps)
    d_PD = 0.5 * (1.0 - u_t) * (1.0 - u_v)
    d_CC = torch.sum(torch.abs(p_t - p_v), dim=1, keepdim=True)
    return (d_PD * d_CC).mean()


# Sqrt-inverse class weights (gentler than full inverse-frequency)
def get_class_weights_tensor(train_df):
    counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    if CFG.CLASS_WEIGHT_MODE == "sqrt_inv":
        weights = np.sqrt(counts.sum() / (CFG.NUM_CLASSES * counts))
    elif CFG.CLASS_WEIGHT_MODE == "inv":
        weights = counts.sum() / (CFG.NUM_CLASSES * counts)
    else:
        return None
    return torch.tensor(weights, dtype=torch.float32, device=CFG.DEVICE)


print("\\u2705 EvidentialLossV3 (digamma + label smoothing + sqrt weights) defined.")
'''

# Cell 11 (UCE) -- UNCHANGED from v2

# ──────────────────────────────────────────────
# Cell 12: Training loop (v3: early stopping + per-seed + gate stats)
# ──────────────────────────────────────────────
cells[12].source = '''# ============================================================
# TRAINING ONE SEED (v3: early stopping, gate statistics, per-seed tracking)
# ============================================================

def train_one_seed(seed, model_kwargs=None):
    """Train the ADEF v3 model for one random seed.
    Returns: (model, best_state_dict, best_val_macro_f1, history_dict)
    """
    set_seed(seed)
    torch.cuda.empty_cache()

    # Build model
    if model_kwargs is None:
        model_kwargs = {}
    model = create_model_v3(model_kwargs)

    # Class weights
    class_weights = None
    if CFG.USE_CLASS_WEIGHTS:
        class_weights = get_class_weights_tensor(train_df)
        print(f"  Class weights (Neg/Neu/Pos): {class_weights.cpu().numpy().round(3).tolist()}")

    criterion = EvidentialLossV3(
        num_classes=CFG.NUM_CLASSES,
        annealing_epochs=CFG.ANNEALING_EPOCHS,
        class_weights=class_weights,
        loss_type=CFG.EDL_LOSS_TYPE,
        label_smoothing=CFG.LABEL_SMOOTHING
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.SCHED_TMAX)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1_macro": [], "val_f1_macro": [],
        "train_conflict": [], "val_conflict": [],
        "train_gate": [], "val_gate": []
    }

    best_score = 0.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_conflict_sum = 0.0
        train_gate_sum = 0.0
        train_preds, train_labels_li = [], []

        pbar = tqdm(train_loader, desc=f"Seed {seed} E{epoch}/{CFG.EPOCHS} [Train]", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(CFG.DEVICE)
            attention_mask = batch["attention_mask"].to(CFG.DEVICE)
            images = batch["image"].to(CFG.DEVICE)
            labels = batch["label"].to(CFG.DEVICE)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, images)

            # L_sup = L(alpha_t) + L(alpha_v) + L(alpha_c)
            L_sup = (criterion(out["alpha_t"], labels, epoch)
                     + criterion(out["alpha_v"], labels, epoch)
                     + criterion(out["alpha_c"], labels, epoch))

            # L_fused: directly supervise the fused decision
            alpha_f = opinion_to_dirichlet(out["b_fusion"], out["u_fusion"],
                                           CFG.NUM_CLASSES, CFG.U_MIN)
            L_fused = criterion(alpha_f, labels, epoch)

            # L_con: semantic conflict (gamma = 0.1)
            L_con = semantic_conflict_loss(out["alpha_t"], out["alpha_v"], CFG.NUM_CLASSES)

            loss = L_sup + CFG.LAMBDA_FUSED * L_fused + CFG.GAMMA * L_con

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            train_conflict_sum += out["K_tv"].mean().item()
            train_gate_sum += out["gate"].mean().item()

            preds = torch.argmax(out["p_final"], dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_li.extend(labels.cpu().numpy())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "K_tv": f"{out['K_tv'].mean().item():.3f}",
                "g": f"{out['gate'].mean().item():.3f}"
            })

        scheduler.step()

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0.0
        val_conflict_sum = 0.0
        val_gate_sum = 0.0
        val_preds, val_labels_li = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Seed {seed} E{epoch}/{CFG.EPOCHS} [Val]", leave=False):
                input_ids = batch["input_ids"].to(CFG.DEVICE)
                attention_mask = batch["attention_mask"].to(CFG.DEVICE)
                images = batch["image"].to(CFG.DEVICE)
                labels = batch["label"].to(CFG.DEVICE)

                out = model(input_ids, attention_mask, images)

                L_sup = (criterion(out["alpha_t"], labels, epoch)
                         + criterion(out["alpha_v"], labels, epoch)
                         + criterion(out["alpha_c"], labels, epoch))
                alpha_f = opinion_to_dirichlet(out["b_fusion"], out["u_fusion"],
                                               CFG.NUM_CLASSES, CFG.U_MIN)
                L_fused = criterion(alpha_f, labels, epoch)
                L_con = semantic_conflict_loss(out["alpha_t"], out["alpha_v"], CFG.NUM_CLASSES)
                loss = L_sup + CFG.LAMBDA_FUSED * L_fused + CFG.GAMMA * L_con

                val_loss += loss.item()
                val_conflict_sum += out["K_tv"].mean().item()
                val_gate_sum += out["gate"].mean().item()

                preds = torch.argmax(out["p_final"], dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_li.extend(labels.cpu().numpy())

        n_train = len(train_loader)
        n_val = len(val_loader)

        # Epoch metrics
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_val
        train_acc = accuracy_score(train_labels_li, train_preds)
        val_acc = accuracy_score(val_labels_li, val_preds)
        train_f1m = f1_score(train_labels_li, train_preds, average="macro")
        val_f1m = f1_score(val_labels_li, val_preds, average="macro")
        avg_train_conflict = train_conflict_sum / n_train
        avg_val_conflict = val_conflict_sum / n_val
        avg_train_gate = train_gate_sum / n_train
        avg_val_gate = val_gate_sum / n_val

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1_macro"].append(train_f1m)
        history["val_f1_macro"].append(val_f1m)
        history["train_conflict"].append(avg_train_conflict)
        history["val_conflict"].append(avg_val_conflict)
        history["train_gate"].append(avg_train_gate)
        history["val_gate"].append(avg_val_gate)

        print(f"Seed {seed} E{epoch:2d}/{CFG.EPOCHS} | "
              f"TL: {avg_train_loss:.4f} VL: {avg_val_loss:.4f} | "
              f"TA: {train_acc:.4f} VA: {val_acc:.4f} | "
              f"VF1m: {val_f1m:.4f} | "
              f"K_tv: {avg_val_conflict:.4f} Gate: {avg_val_gate:.3f}")

        # Early stopping based on val Macro-F1
        score = val_f1m
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
            print(f"  * New best! Val F1m: {best_score:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= CFG.EARLY_STOP_PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improvement for {CFG.EARLY_STOP_PATIENCE} epochs)")
            break

    history["best_epoch"] = best_epoch
    history["best_score"] = best_score
    return model, best_state, best_score, history


print("\\u2705 train_one_seed() with early stopping defined.")
'''

# ──────────────────────────────────────────────
# Cell 13: Test evaluation (v3: gate statistics)
# ──────────────────────────────────────────────
cells[13].source = '''# ============================================================
# TEST SET EVALUATION (v3: gate statistics)
# ============================================================

def evaluate_test(model, test_loader):
    """Evaluate on test set. Returns dict of all metrics."""
    model.eval()
    all_preds, all_labels_li = [], []
    all_uncertainties, all_conflicts, all_gates = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation", leave=False):
            input_ids = batch["input_ids"].to(CFG.DEVICE)
            attention_mask = batch["attention_mask"].to(CFG.DEVICE)
            images = batch["image"].to(CFG.DEVICE)
            labels = batch["label"].to(CFG.DEVICE)

            out = model(input_ids, attention_mask, images)

            preds = torch.argmax(out["p_final"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_li.extend(labels.cpu().numpy())
            all_uncertainties.extend(out["u_fusion"].squeeze(1).cpu().numpy())
            all_conflicts.extend(out["K_tv"].squeeze(1).cpu().numpy())
            all_gates.extend(out["gate"].squeeze(1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels_li = np.array(all_labels_li)
    all_uncertainties = np.array(all_uncertainties)
    all_conflicts = np.array(all_conflicts)
    all_gates = np.array(all_gates)

    acc = accuracy_score(all_labels_li, all_preds)
    f1w = f1_score(all_labels_li, all_preds, average="weighted")
    f1m = f1_score(all_labels_li, all_preds, average="macro")
    f1_per = f1_score(all_labels_li, all_preds, average=None)
    uce = compute_uce(all_preds, all_labels_li, all_uncertainties, CFG.UCE_BINS)

    # Gate / routing statistics
    gate_mean = all_gates.mean()
    gate_median = np.median(all_gates)
    ds_like_pct = (all_gates < 0.5).mean() * 100    # g < 0.5 = agreement, Dempster dominates
    conflict_pct = (all_gates >= 0.5).mean() * 100   # g >= 0.5 = conflict-aware blending dominates

    # K_tv statistics
    k_tv_mean = all_conflicts.mean()
    k_tv_std = all_conflicts.std()
    k_tv_quantiles = np.percentile(all_conflicts, [50, 80, 90, 95, 99])

    # Uncertainty analysis
    correct_mask = all_preds == all_labels_li
    u_correct = all_uncertainties[correct_mask].mean() if correct_mask.sum() > 0 else 0
    u_incorrect = all_uncertainties[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

    results = {
        "acc": acc, "f1_weighted": f1w, "f1_macro": f1m,
        "f1_neg": f1_per[0], "f1_neu": f1_per[1], "f1_pos": f1_per[2],
        "uce": uce,
        "gate_mean": gate_mean, "gate_median": gate_median,
        "ds_like_pct": ds_like_pct, "conflict_pct": conflict_pct,
        "k_tv_mean": k_tv_mean, "k_tv_std": k_tv_std,
        "k_tv_quantiles": k_tv_quantiles,
        "u_correct": u_correct, "u_incorrect": u_incorrect,
        "preds": all_preds, "labels": all_labels_li,
        "uncertainties": all_uncertainties, "conflicts": all_conflicts,
        "gates": all_gates
    }
    return results


def print_results(res, title="RESULTS", tau_ref=None):
    print(f"\\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {res['acc']:.4f}")
    print(f"  F1 (Weighted):     {res['f1_weighted']:.4f}")
    print(f"  F1 (Macro):        {res['f1_macro']:.4f}")
    print(f"  F1 (Negative):     {res['f1_neg']:.4f}")
    print(f"  F1 (Neutral):      {res['f1_neu']:.4f}")
    print(f"  F1 (Positive):     {res['f1_pos']:.4f}")
    print(f"  UCE:               {res['uce']:.4f}")
    print(f"{'='*60}")
    print(f"  Soft Gate Statistics:")
    print(f"    Mean / Median:   {res['gate_mean']:.4f} / {res['gate_median']:.4f}")
    print(f"    g < 0.5 (DS-like):  {res['ds_like_pct']:.1f}%")
    print(f"    g >= 0.5 (conflict): {res['conflict_pct']:.1f}%")
    print(f"  Conflict K_tv:")
    print(f"    Mean:            {res['k_tv_mean']:.4f} +/- {res['k_tv_std']:.4f}")
    print(f"    p50/p80/p90/p95: {np.round(res['k_tv_quantiles'][:4], 4).tolist()}")
    print(f"  Uncertainty:")
    print(f"    u(correct):      {res['u_correct']:.4f}")
    print(f"    u(incorrect):    {res['u_incorrect']:.4f}")
    print(f"{'='*60}")
    print(f"\\nClassification Report:")
    print(classification_report(
        res["labels"], res["preds"],
        target_names=["Negative", "Neutral", "Positive"]
    ))


print("\\u2705 evaluate_test() and print_results() defined.")
'''

# ──────────────────────────────────────────────
# Cell 14: Main experiment (3-seed E1) + viz
# ──────────────────────────────────────────────
cells[14].source = '''# ============================================================
# EXPERIMENT E1: ADEF v3 Full (3 seeds, FILTER_CONFLICT_PAIRS=True)
# ============================================================

print("=" * 60)
print("  EXPERIMENT E1: ADEF v3 Full (3 seeds)")
print(f"  FILTER_CONFLICT_PAIRS = {CFG.FILTER_CONFLICT_PAIRS}")
print(f"  GATE_TYPE = {CFG.GATE_TYPE},  EDL_LOSS_TYPE = {CFG.EDL_LOSS_TYPE}")
print(f"  COATTN_LAYERS = {CFG.COATTN_LAYERS},  OPINION_DROP = {CFG.OPINION_DROP_PROB}")
print(f"  DISCOUNTING = {CFG.USE_DISCOUNTING},  GAMMA = {CFG.GAMMA}")
print("=" * 60)

all_seed_results = []
all_seed_histories = []
learned_params = []

for seed in CFG.SEEDS:
    print(f"\\n{'#'*50}")
    print(f"  Seed {seed}")
    print(f"{'#'*50}")

    model_trained, best_state, best_score, history = train_one_seed(seed)

    # Evaluate with best checkpoint
    model_eval = create_model_v3()
    model_eval.load_state_dict(best_state)
    res = evaluate_test(model_eval, test_loader)

    all_seed_results.append(res)
    all_seed_histories.append(history)

    # Collect learned gate parameters
    if CFG.GATE_TYPE == "parametric":
        tau_learned = 0.01 + 0.49 * torch.sigmoid(model_eval.adef.gate.theta_tau).item()
        s_learned = F.softplus(model_eval.adef.gate.theta_temp).item() + 1e-4
        r_vals = []
        if CFG.USE_DISCOUNTING:
            r_vals = [
                torch.sigmoid(model_eval.adef.discounter_t.raw_r).item(),
                torch.sigmoid(model_eval.adef.discounter_v.raw_r).item(),
                torch.sigmoid(model_eval.adef.discounter_c.raw_r).item()
            ]
        learned_params.append({"tau": tau_learned, "s": s_learned, "r": r_vals, "seed": seed})

    print(f"  Seed {seed}: Best val F1m={best_score:.4f} | Test F1m={res['f1_macro']:.4f} | Epochs={history['best_epoch']}")

    del model_trained, model_eval
    torch.cuda.empty_cache()

# ---- Aggregate over seeds ----
print(f"\\n{'='*60}")
print("  AGGREGATED 3-SEED RESULTS (mean +/- std)")
print(f"{'='*60}")
agg_keys = ["acc", "f1_weighted", "f1_macro", "f1_neg", "f1_neu", "f1_pos", "uce",
            "gate_mean", "k_tv_mean", "u_correct", "u_incorrect"]
for key in agg_keys:
    vals = [r[key] for r in all_seed_results]
    mu, std = np.mean(vals), np.std(vals)
    print(f"  {key:<25s}: {mu:.4f} +/- {std:.4f}")

# ---- Learned parameters ----
if learned_params:
    print(f"\\n  Learned Gate Parameters:")
    for lp in learned_params:
        print(f"    Seed {lp['seed']}: tau={lp['tau']:.4f}  s={lp['s']:.4f}", end="")
        if lp['r']:
            print(f"  r[t,v,c]=[{lp['r'][0]:.3f}, {lp['r'][1]:.3f}, {lp['r'][2]:.3f}]")
        else:
            print()

print(f"\\n  Best epochs: {[h['best_epoch'] for h in all_seed_histories]}")

# Store last seed result for visualization
best_res = all_seed_results[-1]
best_hist = all_seed_histories[-1]
print("\\n\\u2705 E1 complete.")


# ============================================================
# VISUALIZATION (v3: gate distribution replaces tau-based routing)
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle("ADEF v3 -- Learned Soft-Gate Fusion & Deep Gated Co-Attention", fontsize=16, fontweight="bold")

# 1. Loss Curve
axes[0, 0].plot(best_hist["train_loss"], label="Train", marker="o", markersize=3)
axes[0, 0].plot(best_hist["val_loss"], label="Val", marker="s", markersize=3)
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss Curve"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy Curve
axes[0, 1].plot(best_hist["train_acc"], label="Train", marker="o", markersize=3)
axes[0, 1].plot(best_hist["val_acc"], label="Val", marker="s", markersize=3)
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Accuracy Curve"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# 3. Macro-F1 Curve
axes[0, 2].plot(best_hist["train_f1_macro"], label="Train", marker="o", markersize=3)
axes[0, 2].plot(best_hist["val_f1_macro"], label="Val", marker="s", markersize=3)
axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("Macro-F1")
axes[0, 2].set_title("Macro-F1 (selection metric)"); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

# 4. Conflict K_tv + Soft Gate
ax_cg = axes[0, 3]
ax_cg.plot(best_hist["train_conflict"], label="Train K_tv", marker="o", markersize=3, color="orange")
ax_cg.plot(best_hist["val_conflict"], label="Val K_tv", marker="s", markersize=3, color="red")
ax_cg.plot(best_hist["val_gate"], label="Val Gate (g)", marker="^", markersize=3, color="purple")
ax_cg.set_xlabel("Epoch"); ax_cg.set_ylabel("Value")
ax_cg.set_title("Conflict (K_tv) & Soft Gate (g)"); ax_cg.legend(); ax_cg.grid(True, alpha=0.3)

# 5. Confusion Matrix
cm = confusion_matrix(best_res["labels"], best_res["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"],
            ax=axes[1, 0])
axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Actual")
axes[1, 0].set_title("Confusion Matrix")

# 6. Uncertainty Distribution (Correct vs Incorrect)
correct_mask = best_res["preds"] == best_res["labels"]
axes[1, 1].hist(best_res["uncertainties"][correct_mask], bins=30, alpha=0.6,
                label=f"Correct (n={correct_mask.sum()})", color="green", density=True)
if (~correct_mask).sum() > 0:
    axes[1, 1].hist(best_res["uncertainties"][~correct_mask], bins=30, alpha=0.6,
                    label=f"Incorrect (n={(~correct_mask).sum()})", color="red", density=True)
axes[1, 1].set_xlabel("Uncertainty (u)"); axes[1, 1].set_ylabel("Density")
axes[1, 1].set_title("Uncertainty Distribution"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

# 7. Per-Class F1 Scores
class_names = ["Negative", "Neutral", "Positive"]
f1_per = [best_res["f1_neg"], best_res["f1_neu"], best_res["f1_pos"]]
bars = axes[1, 2].bar(class_names, f1_per, color=["#e74c3c", "#3498db", "#2ecc71"])
axes[1, 2].set_xlabel("Class"); axes[1, 2].set_ylabel("F1 Score")
axes[1, 2].set_title("Per-Class F1 Score"); axes[1, 2].set_ylim(0, 1)
for bar, val in zip(bars, f1_per):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
axes[1, 2].grid(True, alpha=0.3, axis="y")

# 8. Soft Gate Distribution
axes[1, 3].hist(best_res["gates"], bins=30, alpha=0.7, color="purple")
axes[1, 3].axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="g = 0.5")
axes[1, 3].axvline(x=best_res["gate_mean"], color="blue", linestyle=":", alpha=0.7,
                   label=f"Mean = {best_res['gate_mean']:.3f}")
axes[1, 3].set_xlabel("Soft Gate (g)"); axes[1, 3].set_ylabel("Count")
axes[1, 3].set_title(f"Soft Gate Distribution\\ng<0.5: {best_res['ds_like_pct']:.0f}%  g>=0.5: {best_res['conflict_pct']:.0f}%")
axes[1, 3].legend(); axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("adef_v3_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\\u2705 Visualization complete. Saved as adef_v3_results.png")
'''

# ──────────────────────────────────────────────
# Add Cell 15: Ablation experiment runner
# ──────────────────────────────────────────────
ablation_cell = nbf.v4.new_code_cell('''# ============================================================
# ABLATION EXPERIMENTS (E2-E8)
# ============================================================
# Quick ablations -- 1 seed each, compares Macro-F1.
# Saved for manual exploration; not required for thesis proof.

def run_ablation(name, overrides, seed=42):
    set_seed(seed)
    torch.cuda.empty_cache()
    print(f"\\n{'='*50}")
    print(f"  {name}")
    print(f"  Overrides: {overrides}")
    print(f"{'='*50}")

    _, best_state, best_score, hist = train_one_seed(seed, overrides)

    model_eval = create_model_v3(overrides)
    model_eval.load_state_dict(best_state)
    res = evaluate_test(model_eval, test_loader)
    print_results(res, title=name)

    del model_eval
    torch.cuda.empty_cache()
    return res

experiments = []

# E2: Hard routing (disable discounting + dropout, parametric gate still soft)
# True hard routing needs gate_type="hard" which doesn"t exist --
# this is V2-style: single-layer co-attention, no discounting, no dropout
res = run_ablation("E2: v2-style (1-layer, no disc, no drop)", {
    "coattn_layers": 1, "use_discounting": False, "opinion_drop_prob": 0.0,
    "gate_type": "parametric"
})
experiments.append(["E2: v2-style", res["f1_macro"], res["f1_neu"], res["acc"]])

# E3: No vacuous dropout
res = run_ablation("E3: No vacuous dropout", {"opinion_drop_prob": 0.0})
experiments.append(["E3: No vac drop", res["f1_macro"], res["f1_neu"], res["acc"]])

# E4: 1-layer co-attention
res = run_ablation("E4: 1-layer co-attention", {"coattn_layers": 1})
experiments.append(["E4: 1-layer CA", res["f1_macro"], res["f1_neu"], res["acc"]])

# E5: MLP gate
res = run_ablation("E5: MLP gate", {"gate_type": "mlp"})
experiments.append(["E5: MLP gate", res["f1_macro"], res["f1_neu"], res["acc"]])

# E6: SOS loss
old_loss = CFG.EDL_LOSS_TYPE
CFG.EDL_LOSS_TYPE = "sos"
res = run_ablation("E6: SOS loss", {})
CFG.EDL_LOSS_TYPE = old_loss
experiments.append(["E6: SOS loss", res["f1_macro"], res["f1_neu"], res["acc"]])

# E7: No discounting
res = run_ablation("E7: No discounting", {"use_discounting": False})
experiments.append(["E7: No discount", res["f1_macro"], res["f1_neu"], res["acc"]])

# E8: Both loss types
CFG.EDL_LOSS_TYPE = "both"
res = run_ablation("E8: Both (digamma+sos)", {})
CFG.EDL_LOSS_TYPE = "digamma"
experiments.append(["E8: Both losses", res["f1_macro"], res["f1_neu"], res["acc"]])

# Summary table
print(f"\\n{'='*60}")
print("  ABLATION SUMMARY (1-seed each)")
print(f"{'='*60}")
print(f"  {'Ablation':<25s} | {'Macro F1':>8s} | {'Neu F1':>6s} | {'Acc':>6s}")
print("-" * 58)
for row in experiments:
    print(f"  {row[0]:<25s} | {row[1]:>8.4f} | {row[2]:>6.4f} | {row[3]:>6.4f}")
print(f"{'='*60}")
print("\\n\\u2705 Ablation experiments complete.")
''')
cells.append(ablation_cell)

# ──────────────────────────────────────────────
# FINAL: Write v3 notebook
# ──────────────────────────────────────────────
nb.cells = cells
nbf.write(nb, DST)
print(f"\\nV3 notebook written to: {DST}")
print(f"Total cells: {len(cells)}")

