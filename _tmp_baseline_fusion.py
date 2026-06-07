
# ============================================================
# BASELINE FUSION MODEL
# Architecture: Self-Attention + Concatenation (Late Fusion)
#
# FUSION MECHANISM:
#   1. Text  -> RoBERTa (pooled [CLS] output,  dim=768)
#   2. Image -> ViT-B/16 (global [CLS] token,  dim=768)
#   3. Concat the two independent representations  -> 1536-dim
#   4. MLP classifier: Linear(1536->256) -> ReLU -> Dropout -> Linear(256->3)
#
# This is pure late fusion -- no cross-modal interaction.
# ============================================================

class BaselineFusionModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        # --- Text Encoder: RoBERTa-base ---
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self._freeze_roberta_layers(num_unfreeze=4)

        # --- Image Encoder: ViT-B/16 ---
        self.image_encoder = vit_b_16(pretrained=True)
        self.image_encoder.heads = nn.Identity()
        self._freeze_vit_blocks(num_unfreeze=4)

        # --- Late Fusion: Concat -> MLP ---
        text_dim = self.text_encoder.config.hidden_size   # 768
        image_dim = 768  # ViT-B/16 hidden_dim
        fused_dim = text_dim + image_dim                   # 1536
        hidden_dim = 256

        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _freeze_roberta_layers(self, num_unfreeze=4):
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

        total_layers = self.text_encoder.config.num_hidden_layers  # 12
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < total_layers - num_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def _freeze_vit_blocks(self, num_unfreeze=4):
        for param in self.image_encoder.conv_proj.parameters():
            param.requires_grad = False

        total_blocks = len(self.image_encoder.encoder.layers)  # 12
        for i, block in enumerate(self.image_encoder.encoder.layers):
            if i < total_blocks - num_unfreeze:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, image):
        # RoBERTa: pooled [CLS] output  (B, 768)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output

        # ViT: global [CLS] token  (B, 768)
        image_features = self.image_encoder(image)

        # Late Fusion: concatenate then classify
        fused = torch.cat([text_features, image_features], dim=1)  # (B, 1536)
        logits = self.fusion(fused)                                 # (B, 3)

        return logits


model = BaselineFusionModel(num_classes=3).to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable_params:,} / {total_params:,}  ({100*trainable_params/total_params:.1f}%)')
