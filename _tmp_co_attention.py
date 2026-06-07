
# ============================================================
# CO-ATTENTION FUSION MODEL
# Architecture: Bidirectional Affinity Matrix Co-Attention
#
# FUSION MECHANISM (Co-Attention):
#   1. Text  -> RoBERTa last_hidden_state      (B, T, 768)
#   2. Image -> ViT-B/16 patch tokens           (B, P, 768)
#   3. Project both to shared space d=512 via Linear layers
#   4. Affinity Matrix: C = tanh(X_t @ W_b @ X_v^T)  (B, T, P)
#   5. Bidirectional Co-Attention:
#      - Text attends over Image:  H_t_co = softmax(C) @ X_v
#      - Image attends over Text:  H_v_co = softmax(C^T) @ X_t
#   6. Mean pool both -> concat -> MLP classifier (3 classes)
# ============================================================

class CoAttentionFusionModel(nn.Module):
    def __init__(self, num_classes=3, joint_dim=512, dropout=0.3):
        super().__init__()

        # --- Text Encoder: RoBERTa-base (sequence output) ---
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.text_hidden = self.text_encoder.config.hidden_size  # 768
        self._freeze_roberta_layers(num_unfreeze=4)

        # --- Image Encoder: ViT-B/16 (patch sequence) ---
        vit = vit_b_16(pretrained=True)
        self.vit_conv_proj = vit.conv_proj
        self.vit_class_token = vit.class_token
        self.vit_encoder = vit.encoder
        self.image_hidden = vit.hidden_dim  # 768
        self._freeze_vit_blocks(num_unfreeze=4)

        # --- Projection to joint space ---
        self.text_proj = nn.Linear(self.text_hidden, joint_dim)
        self.image_proj = nn.Linear(self.image_hidden, joint_dim)

        # --- Co-Attention: bilinear affinity weight ---
        self.W_b = nn.Parameter(torch.randn(joint_dim, joint_dim) * 0.02)

        # --- Classifier ---
        hidden_dim = 256
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 2, hidden_dim),
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
    # IMAGE PATCH ENCODING
    # ------------------------------------------------
    def encode_image_patches(self, x):
        # x: (B, 3, 224, 224)
        n = x.shape[0]
        x = self.vit_conv_proj(x)              # (B, 768, 14, 14)
        x = x.flatten(2).permute(0, 2, 1)      # (B, 196, 768)
        cls = self.vit_class_token.expand(n, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)         # (B, 197, 768)
        x = self.vit_encoder(x)                # (B, 197, 768)  incl. pos_embed
        return x[:, 1:, :]                     # patch tokens only (B, 196, 768)

    # ------------------------------------------------
    # FORWARD
    # ------------------------------------------------
    def forward(self, input_ids, attention_mask, image):
        B = input_ids.shape[0]

        # 1. Unimodal encodings
        text_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        X_t = text_out.last_hidden_state   # (B, T, 768)
        X_v = self.encode_image_patches(image)  # (B, P, 768)

        # 2. Project to joint space d=512
        X_t_proj = self.text_proj(X_t)     # (B, T, d)
        X_v_proj = self.image_proj(X_v)    # (B, P, d)

        # 3. Affinity Matrix: C[i,j] = affinity(text_token_i, image_patch_j)
        #    C = tanh( X_t_proj @ W_b @ X_v_proj^T )  -- bilinear form
        X_t_w = X_t_proj @ self.W_b               # (B, T, d)
        C = torch.tanh(X_t_w @ X_v_proj.transpose(-2, -1))  # (B, T, P)

        # 4. Bidirectional Co-Attention
        #    Text attends over Image patches:
        A_t2v = torch.softmax(C, dim=-1)          # (B, T, P)
        H_t_co = A_t2v @ X_v_proj                 # (B, T, d)  text enriched with visual info

        #    Image attends over Text tokens:
        A_v2t = torch.softmax(C.transpose(-2, -1), dim=-1)  # (B, P, T)
        H_v_co = A_v2t @ X_t_proj                 # (B, P, d)  image enriched with textual info

        # 5. Pool co-attended sequences -> concat -> classify
        t_pooled = H_t_co.mean(dim=1)  # (B, d)
        v_pooled = H_v_co.mean(dim=1)  # (B, d)
        fused = torch.cat([t_pooled, v_pooled], dim=1)  # (B, 2d)
        logits = self.classifier(fused)                    # (B, 3)

        return logits


model = CoAttentionFusionModel(num_classes=3).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)')
