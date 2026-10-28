# Models 5-11 Pre-training Readiness Report

## MODEL 5 — APPLE
- **Classes**: 4
- **Train**: 2536
- **Validation**: 316
- **Test**: 319
- **Mean**: [0.07347409, 0.10038915, 0.0658901]
- **Std**: [0.1349588, 0.17648672, 0.12671633]
- **Class weights**: [0.59785894, 0.96607625, 0.97383602, 1.46222879]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## MODEL 6 — CHERRY
- **Classes**: 2
- **Train**: 1524
- **Validation**: 190
- **Test**: 192
- **Mean**: [0.14351691, 0.18846236, 0.11188254]
- **Std**: [0.18727443, 0.24050464, 0.15725942]
- **Class weights**: [1.05197732, 0.94802268]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## MODEL 7 — CORN
- **Classes**: 4
- **Train**: 3080
- **Validation**: 384
- **Test**: 388
- **Mean**: [0.13903937, 0.16719028, 0.10428933]
- **Std**: [0.20518272, 0.23928076, 0.16130961]
- **Class weights**: [0.87366948, 1.31511307, 0.86259825, 0.94861921]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## MODEL 8 — GRAPE
- **Classes**: 4
- **Train**: 3248
- **Validation**: 405
- **Test**: 409
- **Mean**: [0.13681938, 0.17084016, 0.08285125]
- **Std**: [0.1825235, 0.21670182, 0.11859776]
- **Class weights**: [1.43983018, 0.86155623, 0.79596149, 0.9026521]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## MODEL 9 — PEACH
- **Classes**: 2
- **Train**: 2125
- **Validation**: 265
- **Test**: 267
- **Mean**: [0.10522229, 0.10649589, 0.04322645]
- **Std**: [0.20703991, 0.19445668, 0.08863733]
- **Class weights**: [1.43271484, 0.56728516]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## MODEL 10 — PEPPER
- **Classes**: 2
- **Train**: 1979
- **Validation**: 246
- **Test**: 250
- **Mean**: [0.1214166, 0.16457595, 0.0872918]
- **Std**: [0.17571826, 0.22456476, 0.13301843]
- **Class weights**: [0.90179046, 1.09820954]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## MODEL 11 — STRAWBERRY
- **Classes**: 2
- **Train**: 1251
- **Validation**: 155
- **Test**: 159
- **Mean**: [0.19913457, 0.21676643, 0.1166481]
- **Std**: [0.23870961, 0.25625631, 0.15102065]
- **Class weights**: [1.21906406, 0.78093594]
- **DataLoader**: PASS
- **ResNet-9 output**: PASS

## CUDA Environment
- **PyTorch**: 2.11.0+cu128
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU
- **Compute capability**: sm_120
- **CUDA forward pass**: PASS
- **AMP**: PASS

## Test Set Lock
Test sets remain completely isolated. No test data was exposed to normalization calculations, augmentation, or inference.