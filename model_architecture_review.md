# Comparative Analysis of Cross-Modal Fusion Architectures
## Comprehensive Code Review

---

## 1. Executive Summary

Four multimodal fusion architectures have been implemented for a **3-class hateful-meme classification task**. All models share identical **encoder backbones** (RoBERTa-base + ViT-B/16), **freezing strategy** (bottom 8 layers frozen, top 4 trainable), and **training pipeline** (AdamW, lr=2e-5, BS=32, 10 epochs). They differ exclusively in **how text and image representations are fused** — progressing from zero cross-modal interaction to a sophisticated two-level hierarchical fusion with gating.

| # | Notebook | Fusion Type | Cross-Modal Interaction |
|---|----------|-------------|-------------------------|
| 1 | `baseline_fusion` | Late Fusion | None (independent CLS tokens concatenated) |
| 2 | `co_attention` | Bidirectional Co-Attention | Full: affinity matrix + dual softmax |
| 3 | `cross_attetion` | Text-as-Query Cross-Attention | One-way: text attends over image (MHA) |
| 4 | `hierarchical_cross_attention` | Two-Level Hierarchical | Bi-level: local MHA + global gate |

---

## 2. Shared Infrastructure

### 2.1 Encoder Backbones

All four models use identical pretrained encoders:

```
Text:  RobertaModel.from_pretrained('roberta-base')
       → output: last_hidden_state (B, T, 768) + pooler_output (B, 768)

Image: vit_b_16(pretrained=True)
       → output: encoder_hidden_states (B, 197, 768)  [CLS + 196 patches]
```

### 2.2 Freezing Strategy

Consistent across all four models — **freeze bottom 8, unfreeze top 4**:

| Component | Frozen | Trainable |
|-----------|--------|-----------|
| Text embeddings | Entirely | — |
| Text layers 1-8 | Entirely | — |
| Text layers 9-12 | — | Entirely |
| Text pooler | — | Entirely |
| ViT conv_proj | Entirely | — |
| ViT blocks 1-8 | Entirely | — |
| ViT blocks 9-12 | — | Entirely |
| ViT class_token, pos_embed | — | Entirely |
| Fusion layers | — | Entirely |

**Rationale**: Pretrained bottom layers capture low-level features (edges, syntax) that transfer well; fine-tuning top layers adapts high-level semantics to the hateful-meme domain.

**Estimated parameter counts** (theoretical):

| Model | Trainable | Total | % Trainable |
|-------|-----------|-------|-------------|
| Baseline | ~57.8M | ~210.8M | ~27.4% |
| Co-Attention | ~58.7M | ~211.7M | ~27.7% |
| Cross-Attention | ~59.4M | ~212.4M | ~28.0% |
| Hierarchical | ~61.0M | ~214.0M | ~28.5% |

### 2.3 Joint Projection Space

Models 2-4 project both modalities into a shared space **d=512** before fusion. Baseline concatenates raw 768-dim CLS outputs directly.

### 2.4 Training Pipeline (Identical Across All)

```
CFG:   BS=32, EPOCHS=10, LR=2e-5, MAX_LEN=128
Opt:   AdamW(model.parameters(), lr=CFG['lr'])
Loss:  nn.CrossEntropyLoss()
Split: 80/10/10 (train/val/test) — stratified
Eval:  accuracy, weighted-F1, classification_report
```

---

## 3. Model 1: Baseline Fusion (`baseline_fusion.ipynb`)

### 3.1 Architecture

```
  Text ──► RoBERTa ──► [CLS] ──┐
                                 ├── Concat ──► MLP(1536→256→3)
  Image ─► ViT-B/16 ─► [CLS] ──┘
```

**Fusion type**: Late Fusion — zero cross-modal interaction.

### 3.2 Forward Pass (Tensor Shapes)

```
input_ids:         (B, 128)          # tokenized text
attention_mask:    (B, 128)
image:             (B, 3, 224, 224)

text_features:     (B, 768)          # RoBERTa pooler_output [CLS]
image_features:    (B, 768)          # ViT head output (with head=Identity)

fused:             (B, 1536)         # torch.cat([text, image], dim=1)
logits:            (B, 3)            # MLP output
```

### 3.3 Mathematical Formulation

$$h = [\text{RoBERTa}_{\text{CLS}}(T) \parallel \text{ViT}_{\text{CLS}}(I)] \in \mathbb{R}^{1536}$$
$$\hat{y} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot h + b_1) + b_2)$$

where $\parallel$ denotes concatenation. No learnable parameters exist between the two modalities — they are independently encoded and fused only at the classifier level.

### 3.4 Code Quality Assessment

**Strengths**:
- Minimal, clean — only 78 lines for the model class
- Freezing helpers are well-structured (`_freeze_roberta_layers`, `_freeze_vit_blocks`)
- Explicit dimension tracking (`text_dim`, `image_dim`, `fused_dim`, `hidden_dim`)
- Uses `pooler_output` — the simplest RoBERTa representation

**Potential issues**:
- `vit_b_16` import uses deprecated `pretrained=True` — should be `weights=ViT_B_16_Weights.IMAGENET1K_V1` in newer `torchvision`
- Sets `image_encoder.heads = nn.Identity()` but some ViT versions use `.head` (singular) — version-dependent
- No residual connection in the MLP — could benefit from LayerNorm before classifier
- No dropout before concatenation — raw CLS tokens may have different activation scales

---

## 4. Model 2: Co-Attention Fusion (`co_attention.ipynb`)

### 4.1 Architecture

```
  Text ──► RoBERTa ──► X_t (B,T,768) ──► Linear(768→512) ──► X_t_proj ──┐
                                                                            │
                                     ┌──────────────────────────────────────┤
                                     │                                      │
                                     ▼                                      ▼
  Image ─► ViT ──► X_v (B,P,768) ──► Linear(768→512) ──► X_v_proj ──►  C = tanh(X_t_proj · W_b · X_v_proj^T)
                                                                          │
                                          ┌───────────────────────────────┤
                                          ▼                               ▼
                              A_t2v = softmax(C)           A_v2t = softmax(C^T)
                              H_t_co = A_t2v · X_v_proj    H_v_co = A_v2t · X_t_proj
                                          │                               │
                                          ▼                               ▼
                                   Mean pool ───────── Concat ───── Mean pool
                                                        │
                                                        ▼
                                                   MLP(1024→256→3)
```

### 4.2 Forward Pass (Tensor Shapes)

```
X_t:               (B, T, 768)       # RoBERTa last_hidden_state, T ≤ 128
X_v:               (B, 196, 768)     # ViT patch tokens (excl. CLS)

X_t_proj:          (B, T, 512)       # text_seq → joint space
X_v_proj:          (B, 196, 512)     # image patches → joint space
W_b:               (512, 512)        # bilinear affinity weight

C:                 (B, T, 196)       # affinity matrix, C[i,j] = sim(text_token_i, patch_j)

A_t2v:             (B, T, 196)       # softmax over patches  (dim=-1)
H_t_co:            (B, T, 512)       # text enriched with visual context

A_v2t:             (B, 196, T)       # softmax over tokens   (dim=-1)
H_v_co:            (B, 196, 512)     # image enriched with textual context

t_pooled + v_pooled: (B, 512) each  → concat → (B, 1024)
logits:            (B, 3)
```

### 4.3 Mathematical Formulation

**Affinity matrix** (bilinear form):
$$C_{ij} = \tanh\left( (X_t^\text{proj} W_b) \cdot (X_v^\text{proj})^T \right)_{ij}$$

This measures the similarity between text token $i$ and image patch $j$ in the projected joint space. The bilinear weight $W_b \in \mathbb{R}^{512 \times 512}$ learns cross-modal interaction patterns.

**Bidirectional attention**:
$$H_t^\text{co} = \text{softmax}_{\text{patches}}(C) \cdot X_v^\text{proj} \quad \in \mathbb{R}^{B \times T \times 512}$$
$$H_v^\text{co} = \text{softmax}_{\text{tokens}}(C^T) \cdot X_t^\text{proj} \quad \in \mathbb{R}^{B \times 196 \times 512}$$

**Fusion**: Both co-attended sequences are mean-pooled and concatenated:
$$h_f = [\text{mean}(H_t^\text{co}) \parallel \text{mean}(H_v^\text{co})] \in \mathbb{R}^{1024}$$

### 4.4 Code Quality Assessment

**Strengths**:
- Clean separation of concerns: `encode_image_patches` method isolates ViT forward logic
- Proper reuse of `vit_conv_proj`, `vit_class_token`, `vit_encoder` from pretrained ViT
- `W_b` parameter initialized with small random values (`* 0.02`) — good practice for bilinear attention
- `torch.tanh` bounds affinity to [-1, 1] — prevents exploding attention weights

**Potential issues**:
- `attention_mask` from text is **not applied** during co-attention — padded text tokens can attend to image patches (and vice versa), injecting noise. Should mask softmax with `~attention_mask.bool()`
- `encode_image_patches` **discards the CLS token** (`x[:, 1:, :]`) — intentional but means global image context is lost
- The bilinear form $X_t W_b X_v^T$ requires $T \times 196$ memory per sample; at T=128 this is 25K entries — manageable but scales quadratically
- No residual connection after co-attention — unlike cross_attetion which has residual + LayerNorm
- Mean pooling treats all tokens equally — a learned pooling (attention or weighted sum) could be more expressive

---

## 5. Model 3: Cross-Attention Fusion (`cross_attetion.ipynb`)

### 5.1 Architecture

```
  Text ──► RoBERTa ──► X_t (B,T,768) ──► Linear(768→512) ──► Q ──┐
                                                                   │
  Image ─► ViT ──► X_v (B,P,768) ──► Linear(768→512) ──► K,V ───┤
                                                                   │
                                          ┌────────────────────────┘
                                          ▼
                              nn.MultiheadAttention(Q, K, V)
                                          │
                                          ▼
                              Residual + LayerNorm
                                          │
                                          ▼
                              Mean pool → MLP(512→256→3)
```

### 5.2 Forward Pass (Tensor Shapes)

```
X_t:               (B, T, 768)
X_v:               (B, 196, 768)

Q (Query):         (B, T, 512)        # from text tokens
K (Key):           (B, 196, 512)      # from image patches
V (Value):         (B, 196, 512)      # same as K

attn_out:          (B, T, 512)        # cross-attention output
+ Q:               (B, T, 512)        # residual connection
→ LayerNorm →      (B, T, 512)        # post-norm

pooled:            (B, 512)           # mean over text tokens
logits:            (B, 3)
```

### 5.3 Mathematical Formulation

Standard multi-head cross-attention with **text as query** and **image as key/value**:

$$\text{head}_h = \text{softmax}\left(\frac{Q W_h^Q \cdot (K W_h^K)^T}{\sqrt{d_k}}\right) \cdot V W_h^V$$

$$\text{CrossAttn}(Q,K,V) = [\text{head}_1 \parallel \dots \parallel \text{head}_8] W^O$$

With residual connection and post-LayerNorm:
$$H^\text{cross} = \text{LayerNorm}(\text{CrossAttn}(Q,K,V) + Q)$$

**Design choice**: V = K means image patches serve as both the attention keys and the information source. Text tokens query which image regions are relevant, and the output is a visual-context-aware text representation.

### 5.4 Code Quality Assessment

**Strengths**:
- Uses PyTorch's `nn.MultiheadAttention` with `batch_first=True` — concise, tested implementation
- Residual + LayerNorm follows Transformer best practices
- `num_heads=8` with `embed_dim=512` gives $d_k = 64$ per head — standard and well-justified
- Simplest dimension path: `512 → 512 → 256 → 3` — no unnecessary intermediate projections
- `attention_mask` from text could be passed to MHA's `key_padding_mask` for proper masking

**Potential issues**:
- **Typo in filename**: `cross_attetion.ipynb` (missing 'n' in "attention") — cosmetic but confusing
- `attention_mask` is again **unused** during cross-attention — padded tokens can attend to image patches
- Image patch attention mask is **not created** — ViT has a fixed 196 patches so no padding, but text has variable length padding that should be masked
- **One-way**: only text attends to image, image never gets enriched by text — limits bidirectionality
- `Value = Key` means the attention output is a weighted sum of projected image patches — the `V` linear projection is shared with `K`, losing one degree of freedom in the projection

---

## 6. Model 4: Hierarchical Cross-Attention (`hierarchical_cross_attention.ipynb`)

### 6.1 Architecture

```
                        LEVEL 1 (Local)                              LEVEL 2 (Global)
  ════════════════════════════════════════     ═══════════════════════════════════════

  Text ──► RoBERTa ──► X_t_seq (B,T,768)      ┌─ T_cls (B,768) ──► Linear ──► T_global (B,512) ──┐
                       │                      │                                                   │
                       ├─► Linear(768→512) ──► Q (B,T,512)                                        │
                       │        │                                                                │
  Image ─► ViT ──► V_cls (B,768)             │                                                │
                   X_v_patches (B,196,768) ──► Linear(768→512) ──► K,V (B,196,512)              │
                                  │           │          │                                        │
                                  │           │  ┌───────┘                                        │
                                  │           │  ▼                                                │
                                  │           │  MHA(Q,K,V) → out                                   │
                                  │           │  Residual + LN                                     │
                                  │           │  Mean pool → local_feat (B,512)                    │
                                  │           │        │                                          │
                                  │           │        │  ┌───────────────────────────────────────┤
                                  │           │        │  │                                       │
                                  │           │        │  │         ┌─ V_cls (B,768) ──► Linear ──► V_global (B,512) ──┘
                                  │           │        │  │         │
                                  │           │        │  │         │
                                  │           │        ▼  ▼         ▼
                                  │           │    gate = σ(Linear([T_global; V_global]))  (B,512)
                                  │           │        │
                                  │           │    local_gated = local_feat * gate          (B,512)
                                  │           │        │
                                  │           │    fused = [T_global; V_global; local_gated] (B,1536)
                                  │           │        │
                                  │           │        ▼
                                  │           │    MLP(1536→256→3)
```

### 6.2 Forward Pass (Tensor Shapes)

```
# Unimodal encodings
X_t_seq:           (B, T, 768)       # full text sequence
T_cls:             (B, 768)          # text [CLS] token
V_cls:             (B, 768)          # image [CLS] token
X_v_patches:       (B, 196, 768)     # image patch tokens

# LEVEL 1 — Local Cross-Attention
Q:                 (B, T, 512)       # text tokens → query
K, V:              (B, 196, 512)     # image patches → key/value
local_out:         (B, T, 512)       # MHA output
  + Q residual:    (B, T, 512)
  → LayerNorm:     (B, T, 512)
  → mean pool:     (B, 512)          # local_feat

# LEVEL 2 — Global Gate
T_global:          (B, 512)          # projected text CLS
V_global:          (B, 512)          # projected image CLS
gate_input:        (B, 1024)         # concat[T_global, V_global]
gate:              (B, 512)          # σ(Linear(gate_input))
local_gated:       (B, 512)          # gate * local_feat

# Final Fusion
fused:             (B, 1536)         # [T_global; V_global; local_gated]
logits:            (B, 3)
```

### 6.3 Mathematical Formulation

**Level 1 — Local Cross-Attention**:
Identical to Model 3 (cross-attention), producing per-text-token representations enriched with relevant image patch information. Mean-pooled to a single vector:

$$\text{local\_feat} = \frac{1}{T} \sum_{i=1}^{T} \text{CrossAttn}(Q_i, K, V) \in \mathbb{R}^{512}$$

**Level 2 — Global Gate**:
$$g = \sigma(W_g \cdot [T_{\text{CLS}}^\text{proj} \parallel V_{\text{CLS}}^\text{proj}] + b_g) \in \mathbb{R}^{512}$$

$$\text{local\_gated} = \text{local\_feat} \odot g \quad \text{(element-wise)}$$

The gate $g$ modulates the local cross-attention signal: when global CLS tokens indicate strong alignment, the gate opens; when they disagree, the gate suppresses local features.

**Final fusion**:
$$h_f = [T_{\text{CLS}}^\text{proj} \parallel V_{\text{CLS}}^\text{proj} \parallel \text{local\_gated}] \in \mathbb{R}^{1536}$$

### 6.4 Code Quality Assessment

**Strengths**:
- Most sophisticated architecture — uniquely combines local and global signals
- **Gate mechanism** is learnable and data-driven — the model decides how much local cross-attention to trust
- Four separate projections (`text_seq_proj`, `image_patch_proj`, `text_cls_proj`, `image_cls_proj`) give maximum flexibility
- **Preserves CLS tokens** — unlike co_attention and cross_attetion which discard them
- Good naming conventions: `local_cross_attn`, `local_layer_norm`, `gate_linear`, `local_gated`

**Potential issues**:
- `attention_mask` unused — same issue as all other models
- **Four projection layers** have identical input/output dimensions (768→512) — could be consolidated into fewer layers to reduce parameter count
- Gate uses `sigmoid` → [0,1] per element — this is a **fine-grained** gate (each of 512 dims is independently gated). A scalar gate (`sigmoid(Linear(1024→1))`) would be simpler but less expressive
- **No learnable interaction** between T_global and V_global beyond concatenation in the gate — could add a simple dot-product or bilinear term
- Three separate inputs to final classifier may have different scales — a LayerNorm before the MLP could help stabilize training

---

## 7. Comparative Analysis

### 7.1 Cross-Modal Interaction Depth

```
Model              Interaction    Bidirectional?   Granularity
─────────────────────────────────────────────────────────────
Baseline           None           —                —
Cross-Attention    One-way        No (T → V)       Token-to-Patch
Co-Attention       Two-way        Yes (T ↔ V)      Token-to-Patch
Hierarchical       Two-way[*]     Partial           Token-to-Patch + CLS Gate

[*] Hierarchical is one-way at Level 1 (T → V), but Level 2
    re-injects bidirectional CLS information through the gate.
```

### 7.2 Computational Complexity

| Model | Fusion FLOPs (per sample) | Memory Bottleneck |
|-------|--------------------------|-------------------|
| Baseline | $O(1)$ — just concat | Negligible |
| Cross-Attention | $O(T \cdot P \cdot d_{head})$ per head | $T \times P$ attention matrix (128×196=25K) |
| Co-Attention | $O(T \cdot P \cdot d)$ + $O(T \cdot d^2)$ bilinear | $T \times P$ affinity ×2 (bidirectional) |
| Hierarchical | $O(T \cdot P \cdot d_{head})$ + $O(d^2)$ gate | $T \times P$ attention + gate linear |

The **affinity matrix** in co-attention and **attention matrix** in cross-attention/hierarchical are both $T \times P$ (max 128×196), which is **very manageable** (~25K entries). The bilinear form in co-attention adds one extra $d^2$ operation (512² = 262K), making it slightly heavier than pure MHA.

### 7.3 Strengths & Weaknesses Matrix

| Aspect | Baseline | Co-Attention | Cross-Attention | Hierarchical |
|--------|----------|--------------|-----------------|--------------|
| **Simplicity** | ++ | - | = | -- |
| **Cross-modal signal** | -- | ++ | = | ++ |
| **Bidirectional** | -- | ++ | - | = |
| **Global context** | + (CLS) | - (no CLS) | - (no CLS) | ++ |
| **Local alignment** | -- | ++ | + | + |
| **Training speed** | ++ | - | = | - |
| **Parameter efficiency** | + | = | + | - |
| **Interpretability** | -- | ++ (affinity viz) | + (attn weights) | + |
| **Risk of overfitting** | Low | Medium | Medium | Higher |

### 7.4 When to Use Each

- **Baseline**: Quick sanity check. If this performs well, the task may not require cross-modal reasoning.
- **Cross-Attention**: When text is the primary modality and image serves as context (e.g., VQA where you query the image with a question).
- **Co-Attention**: When both modalities are equally important and mutual alignment matters (e.g., image-text matching, grounding).
- **Hierarchical**: When you need both fine-grained alignment (word→region) and global coherence (does the overall image match the overall text?).

---

## 8. Cross-Cutting Issues (All Models)

### 8.1 Missing Attention Mask Propagation

**Severity**: Medium-High. All four models receive `attention_mask` but ignore it during fusion.

**Problem**: Padded text tokens (zeros beyond the actual sentence length) participate in attention, injecting meaningless information.

**Fix**: In co-attention and cross-attention, extend the mask to the attention dimensions:

```python
# For co-attention (C shape: B, T, P):
if attention_mask is not None:
    mask_expanded = attention_mask.unsqueeze(-1)  # (B, T, 1)
    C = C.masked_fill(mask_expanded == 0, -1e9)   # before softmax
```

```python
# For cross_attetion MHA:
attn_out, _ = self.cross_attn(Q, K, V, key_padding_mask=(attention_mask == 0))
```

### 8.2 Deprecated `pretrained=True`

**Severity**: Low (warning only). `torchvision` ≥0.14 requires the `weights` parameter:

```python
# Current (deprecated):
vit = vit_b_16(pretrained=True)

# Recommended:
from torchvision.models import ViT_B_16_Weights
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
```

### 8.3 ViT `.heads` vs `.head` Attribute

In co_attention and cross_attetion, the ViT is dismantled into components (`vit_conv_proj`, `vit_encoder`, etc.) so the heads attribute is irrelevant. Only baseline uses `image_encoder.heads = nn.Identity()` — this is version-dependent; some torchvision versions use `.head` (singular). Consider adding a try/except.

### 8.4 No Gradient Scaling / Mixed Precision

Training 211M+ parameter models at BS=32 may exceed GPU memory. Consider:
- `torch.cuda.amp.autocast()` for mixed precision (FP16)
- Gradient accumulation if memory-constrained

### 8.5 Learning Rate Scheduling

All models use a fixed LR. A cosine schedule or warmup could stabilize training, especially for the more complex models (co-attention, hierarchical).

---

## 9. Recommendations

### 9.1 Immediate Fixes (Bug-Level)

1. **Add attention mask everywhere**: Pass `key_padding_mask` to `nn.MultiheadAttention`; mask `C` in co-attention before softmax.
2. **Fix filename typo**: Rename `cross_attetion.ipynb` → `cross_attention.ipynb`.
3. **Update deprecated API**: Replace `pretrained=True` with `weights=` parameter.

### 9.2 Architectural Improvements

| Model | Suggestion |
|-------|-----------|
| Baseline | Add LayerNorm before the MLP to normalize CLS activation scales. |
| Co-Attention | Add residual + LayerNorm after co-attention (like cross_attetion does). Try learned pooling (attention-weighted sum) instead of mean. |
| Cross-Attention | Add a reverse direction (Image attends to Text) and concat both outputs for true bidirectionality. |
| Hierarchical | Try a scalar gate instead of element-wise gate for simpler interpretability. Add cross-attention between T_global and V_global. |

### 9.3 Experimental Priorities

1. **Run baseline first** — establishes the lower bound. If baseline gets >80% accuracy, the task may be simple enough that complex fusion adds noise.
2. **Compare co-attention vs cross-attention** — this is the key ablation: bidirectional affinity vs one-way MHA.
3. **Evaluate hierarchical last** — most parameters, highest risk of overfitting on small datasets.

---

## 10. Conclusion

The four models form a **coherent progression** in fusion complexity:

```
Baseline (Late Fusion)
  │
  ├──► Cross-Attention (MHA: Text→Image)
  │       │
  │       └──► Hierarchical (MHA + Global Gate)
  │
  └──► Co-Attention (Bilinear Affinity: Text↔Image)
```

All share identical infrastructure, making them **directly comparable** — any performance difference is attributable to the fusion mechanism alone. The main code quality issue across all models is the **unused attention mask**, which should be fixed before any training runs.

The architecture selection ultimately depends on the dataset characteristics:
- If memes have **simple text-image relationships** → Baseline or Cross-Attention suffices
- If memes require **mutual grounding** (e.g., "the text says X but the image shows Y") → Co-Attention
- If memes need both **word-level alignment and holistic understanding** → Hierarchical
