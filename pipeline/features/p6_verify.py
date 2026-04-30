# pipeline/features/p6_verify.py
import torch

ckpt = torch.load(
    "checkpoints/transformer_best.pt",
    map_location="cpu",
    weights_only=False   # safe — file was just written by us
)

print(f"feature_cols count:   {len(ckpt['feature_cols'])}")
print(f"n_features in config: {ckpt['config']['n_features']}")
print(f"val_loss:             {ckpt['val_loss']:.6f}")
print(f"best epoch:           {ckpt['epoch']}")
print(f"\nFeature list:")
for i, c in enumerate(ckpt['feature_cols']):
    print(f"  {i+1:02d}. {c}")

assert len(ckpt['feature_cols']) == 46
print("\n✅ Checkpoint verified — 46 features confirmed")