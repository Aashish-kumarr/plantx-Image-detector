# Model Weights Directory

This directory stores the trained PyTorch champion model weights for the PlantDx system.

## Checkpoint Manifest

All model metadata, architecture details, class mappings, normalization parameters, performance metrics, and SHA-256 checksums are documented in:
- [`models_manifest.json`](models_manifest.json)
- [`copied_models.json`](copied_models.json)

## Checkpoint Files

In deployment, this directory contains the following `.pth` state dicts:

| File | Crop | Architecture | Num Classes | Source Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| `model5_apple.pth` | Apple | ResNet-9 | 4 | `checkpoints/model5_apple_final_refine/best_final_refined_model.pth` |
| `model6_cherry.pth` | Cherry | ResNet-9 | 2 | `checkpoints/model6_cherry/best_model.pth` |
| `model7_corn.pth` | Corn | ResNet-9 | 4 | `checkpoints/model7_corn_refine/best_refined_model.pth` |
| `model8_grape.pth` | Grape | ResNet-9 | 4 | `checkpoints/model8_grape/best_model.pth` |
| `model9_peach.pth` | Peach | ResNet-9 | 2 | `checkpoints/model9_peach/best_model.pth` |
| `model10_pepper.pth` | Pepper | ResNet-9 | 2 | `checkpoints/model10_pepper/best_model.pth` |
| `model11_strawberry.pth` | Strawberry | ResNet-9 | 2 | `checkpoints/model11_strawberry/best_model.pth` |

*Note: Models 1–4 weights (`model1_leaf.pth`, `model2_crop.pth`, `model3_potato.pth`, `model4_tomato.pth`) reside under `backend/app/ml/pytorch_models/weights/`.*

Binary weight files (`*.pth`) are excluded from Git version control per standard repository guidelines.
