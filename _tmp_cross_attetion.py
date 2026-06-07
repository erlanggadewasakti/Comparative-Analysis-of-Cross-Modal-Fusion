
# ============================================================
# CROSS-ATTENTION FUSION MODEL
# Architecture: Text-as-Query Cross-Attention over Image Patches
#
# FUSION MECHANISM (Cross-Attention):
#   1. Text  -> RoBERTa last_hidden_state      (B, T, 768)
#   2. Image -> ViT-B/16 patch tokens           (B, P, 768)
#   3. Project both to shared space d=512 via Linear layers
#   4. nn.MultiheadAttention (d=512, heads=8):
#      - Query  (Q): Text tokens     (B, 128, 512)
#      - Key    (K): Image patches   (B, 196, 512)
#      - Value  (V): Image patches   (B, 196, 512)
#      -> Text tokens selectively attend to relevant image regions
#   5. Mean pool attended sequence -> MLP classifier (3 classes)
# ============================================================

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, num_classes=3, joint_dim=512, num_heads=8, dropout=0.3):
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

        # --- Cross-Attention: Text attends over Image ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )

        # --- Layer Normalization ---
        self.layer_norm = nn.LayerNorm(joint_dim)

        # --- Classifier ---
        hidden_dim = 256
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(joint_dim, hidden_dim),
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
        # 1. Unimodal encodings
        text_out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        X_t = text_out.last_hidden_state   # (B, T, 768)
        X_v = self.encode_image_patches(image)  # (B, P, 768)

        # 2. Project to joint space d=512
        Q = self.text_proj(X_t)     # (B, T, d)
        K = self.image_proj(X_v)    # (B, P, d)
        V = K                       # (B, P, d)  Value = Key

        # 3. Cross-Attention: Text tokens attend to Image patches
        attn_out, _ = self.cross_attn(Q, K, V)  # (B, T, d)
        attn_out = self.layer_norm(attn_out + Q) # residual + norm

        # 4. Mean pool over text tokens -> classify
        pooled = attn_out.mean(dim=1)    # (B, d)
        logits = self.classifier(pooled) # (B, 3)

        return logits


model = CrossAttentionFusionModel(num_classes=3).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.1f}%)')
