
# ============================================================
# HIERARCHICAL CROSS-ATTENTION FUSION MODEL
# Architecture: Two-Level Hierarchical Fusion
#
# LEVEL 1 (Local): Cross-Attention between text tokens and
#   image patches -- fine-grained word-to-region alignment.
#   Text tokens (Q) attend over Image patches (K,V) via
#   nn.MultiheadAttention(d=512, heads=8). Residual + Norm,
#   then mean pool -> local_feat (B, 512).
#
# LEVEL 2 (Global): Gated fusion of global CLS tokens with
#   the Level-1 local output. Text [CLS] and Image [CLS]
#   (both projected to 512) compute a sigmoid gate that
#   controls how much local cross-attention signal passes.
#   Final: concat[T_global, V_global, local_gated] -> MLP.
# ============================================================

class HierarchicalFusionModel(nn.Module):
    def __init__(self, num_classes=3, joint_dim=512, num_heads=8, dropout=0.3):
        super().__init__()

        # --- Text Encoder: RoBERTa-base ---
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.text_hidden = self.text_encoder.config.hidden_size  # 768
        self._freeze_roberta_layers(num_unfreeze=4)

        # --- Image Encoder: ViT-B/16 ---
        vit = vit_b_16(pretrained=True)
        self.vit_conv_proj = vit.conv_proj
        self.vit_class_token = vit.class_token
        self.vit_encoder = vit.encoder
        self.image_hidden = vit.hidden_dim  # 768
        self._freeze_vit_blocks(num_unfreeze=4)

        # --- Projections to joint space ---
        self.text_seq_proj  = nn.Linear(self.text_hidden, joint_dim)
        self.image_patch_proj = nn.Linear(self.image_hidden, joint_dim)
        self.text_cls_proj  = nn.Linear(self.text_hidden, joint_dim)
        self.image_cls_proj = nn.Linear(self.image_hidden, joint_dim)

        # --- Level 1: Local Cross-Attention ---
        self.local_cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.local_layer_norm = nn.LayerNorm(joint_dim)

        # --- Level 2: Global Gate ---
        self.gate_linear = nn.Linear(joint_dim * 2, joint_dim)

        # --- Classifier ---
        hidden_dim = 256
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    # ------------------------------------------------
    # FREEZING HELPERS
    # ------------------------------------------------
    def _freeze_roberta_layers(self, num_unfreeze=4):
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        total = self.text_encoder.config.num_hidden_layers  # 12
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < total - num_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def _freeze_vit_blocks(self, num_unfreeze=4):
        for param in self.vit_conv_proj.parameters():
            param.requires_grad = False
        total = len(self.vit_encoder.layers)  # 12
        for i, block in enumerate(self.vit_encoder.layers):
            if i < total - num_unfreeze:
                for param in block.parameters():
                    param.requires_grad = False

    # ------------------------------------------------
    # IMAGE ENCODING: returns both CLS and patches
    # ------------------------------------------------
    def encode_image(self, x):
        # x: (B, 3, 224, 224)
        n = x.shape[0]
        x = self.vit_conv_proj(x)              # (B, 768, 14, 14)
        x = x.flatten(2).permute(0, 2, 1)      # (B, 196, 768)
        cls = self.vit_class_token.expand(n, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)         # (B, 197, 768)
        x = self.vit_encoder(x)                # (B, 197, 768)  incl. pos_embed
        return x[:, 0, :], x[:, 1:, :]         # CLS (B,768), patches (B,196,768)

    # ------------------------------------------------
    # FORWARD
    # ------------------------------------------------
    def forward(self, input_ids, attention_mask, image):
        # 1. Unimodal encodings
        text_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        X_t_seq = text_out.last_hidden_state          # (B, T, 768)
        T_cls   = X_t_seq[:, 0, :]                    # (B, 768)  raw [CLS]
        V_cls, X_v_patches = self.encode_image(image)  # CLS (B,768), patches (B,P,768)

        # ================================================
        # LEVEL 1: Local Cross-Attention
        # Text tokens attend over Image patches
        # ================================================
        Q = self.text_seq_proj(X_t_seq)        # (B, T, d)
        K = self.image_patch_proj(X_v_patches) # (B, P, d)
        V = K                                    # (B, P, d)

        local_out, _ = self.local_cross_attn(Q, K, V)  # (B, T, d)
        local_out   = self.local_layer_norm(local_out + Q)  # residual + norm
        local_feat  = local_out.mean(dim=1)           # (B, d)

        # ================================================
        # LEVEL 2: Global Gated Fusion
        # ================================================
        T_global = self.text_cls_proj(T_cls)    # (B, d)
        V_global = self.image_cls_proj(V_cls)    # (B, d)

        # Gate: global context controls local signal pass-through
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([T_global, V_global], dim=-1))
        )  # (B, d)
        local_gated = local_feat * gate          # (B, d)

        # Fuse all three streams
        fused = torch.cat([T_global, V_global, local_gated], dim=-1)  # (B, 3d)
        logits = self.classifier(fused)  # (B, 3)

        return logits


model = HierarchicalFusionModel(num_classes=3).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)')
