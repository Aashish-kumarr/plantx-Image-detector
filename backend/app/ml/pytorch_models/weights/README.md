# Core Model Checkpoints (Models 1–4)

This directory holds the foundational models in the PlantDx hierarchical inference pipeline:

1. **Model 1**: Leaf vs. Non-Leaf Binary Classifier (`model1_leaf.pth`)
2. **Model 2**: 14-Class Crop Identifier (`model2_crop.pth`)
3. **Model 3**: Potato Disease Classifier (`model3_potato.pth`)
4. **Model 4**: Tomato Disease Classifier (`model4_tomato.pth`)

All models utilize custom ResNet-9 architectures trained from scratch with DSP preprocessing.

Binary weights (`*.pth`) are excluded from Git repository tracking.
