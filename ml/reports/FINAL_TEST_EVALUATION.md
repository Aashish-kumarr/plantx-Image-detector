# PlantDx Final Test Evaluation

This document contains the official final test set evaluation results for all 4 trained/refined models in the PlantDx pipeline.
All evaluations were performed deterministically on held-out, previously untouched test splits.

## Summary Table
| Model | Task | Test Samples | Val Acc | Test Acc | Acc Gap | Val Macro-F1 | Test Macro-F1 | F1 Gap |
|---|---|---|---|---|---|---|---|---|
| MODEL1 | Model 1 ? Leaf vs Non-Leaf | 11,110 | 96.68% | 97.71% | +1.03% | 0.9667 | 0.9771 | +0.0104 |
| MODEL2 | Model 2 ? 14-Class Crop Classifier | 5,431 | 95.01% | 95.12% | +0.11% | 0.9335 | 0.9374 | +0.0039 |
| MODEL3 | Model 3 ? Potato Disease Classifier | 210 | 93.67% | 92.86% | -0.81% | 0.9194 | 0.9139 | -0.0055 |
| MODEL4 | Model 4 ? Tomato Disease Classifier | 1,865 | 93.47% | 93.78% | +0.31% | 0.9131 | 0.9190 | +0.0059 |

## Model 1 ? Leaf vs Non-Leaf (MODEL1) Results
- **Evaluated Checkpoint**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model1\best_model1.pth
- **Test Samples**: 11,110
- **Test Accuracy**: 97.71% (Validation: 96.68%, Gap: +1.03%)
- **Test Macro-F1**: 0.9771 (Validation: 0.9667, Gap: +0.0104)
- **Test Weighted-F1**: 0.9771
- **Test Loss**: 0.0644

### Per-Class Metrics
| Class Name | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| non_leaf | 0.9636 | 0.9928 | 0.9780 | 5,679 |
| leaf | 0.9922 | 0.9608 | 0.9762 | 5,431 |

### Top Confusion Pairs
- Actual leaf -> Predicted non_leaf: **213** images
- Actual non_leaf -> Predicted leaf: **41** images

## Model 2 ? 14-Class Crop Classifier (MODEL2) Results
- **Evaluated Checkpoint**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model2_versions\model2_14class_refine_v1_f1_0.9335_acc_95.01.pth
- **Test Samples**: 5,431
- **Test Accuracy**: 95.12% (Validation: 95.01%, Gap: +0.11%)
- **Test Macro-F1**: 0.9374 (Validation: 0.9335, Gap: +0.0039)
- **Test Weighted-F1**: 0.9514
- **Test Loss**: 0.1481

### Per-Class Metrics
| Class Name | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| apple | 0.8179 | 0.9464 | 0.8775 | 280 |
| blueberry | 0.9933 | 0.9610 | 0.9769 | 154 |
| cherry | 0.8973 | 0.9663 | 0.9306 | 208 |
| corn | 0.9871 | 0.9574 | 0.9720 | 399 |
| grape | 0.9507 | 0.9693 | 0.9599 | 358 |
| orange | 0.9678 | 0.9930 | 0.9803 | 575 |
| peach | 0.9219 | 0.8741 | 0.8973 | 270 |
| pepper | 0.9409 | 0.9560 | 0.9484 | 250 |
| potato | 0.9058 | 0.8238 | 0.8628 | 210 |
| raspberry | 0.7917 | 1.0000 | 0.8837 | 38 |
| soybean | 0.9602 | 0.9602 | 0.9602 | 502 |
| squash | 0.9524 | 0.9816 | 0.9668 | 163 |
| strawberry | 0.9548 | 0.9308 | 0.9427 | 159 |
| tomato | 0.9769 | 0.9523 | 0.9644 | 1,865 |

### Top Confusion Pairs
- Actual potato -> Predicted tomato: **22** images
- Actual peach -> Predicted apple: **20** images
- Actual tomato -> Predicted apple: **17** images
- Actual tomato -> Predicted potato: **17** images
- Actual tomato -> Predicted orange: **12** images
- Actual tomato -> Predicted peach: **9** images
- Actual tomato -> Predicted cherry: **8** images
- Actual tomato -> Predicted grape: **7** images
- Actual tomato -> Predicted pepper: **7** images
- Actual apple -> Predicted peach: **6** images

## Model 3 ? Potato Disease Classifier (MODEL3) Results
- **Evaluated Checkpoint**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model3_versions\model3_potato_refine_v1_f1_0.9194_acc_93.67.pth
- **Test Samples**: 210
- **Test Accuracy**: 92.86% (Validation: 93.67%, Gap: -0.81%)
- **Test Macro-F1**: 0.9139 (Validation: 0.9194, Gap: -0.0055)
- **Test Weighted-F1**: 0.9288
- **Test Loss**: 0.2165

### Per-Class Metrics
| Class Name | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| healthy | 0.8750 | 0.8750 | 0.8750 | 16 |
| early_blight | 0.9773 | 0.9053 | 0.9399 | 95 |
| late_blight | 0.8962 | 0.9596 | 0.9268 | 99 |

### Top Confusion Pairs
- Actual early_blight -> Predicted late_blight: **9** images
- Actual healthy -> Predicted late_blight: **2** images
- Actual late_blight -> Predicted healthy: **2** images
- Actual late_blight -> Predicted early_blight: **2** images

## Model 4 ? Tomato Disease Classifier (MODEL4) Results
- **Evaluated Checkpoint**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model4_versions\model4_tomato_refine_v1_f1_0.9131_acc_93.47.pth
- **Test Samples**: 1,865
- **Test Accuracy**: 93.78% (Validation: 93.47%, Gap: +0.31%)
- **Test Macro-F1**: 0.9190 (Validation: 0.9131, Gap: +0.0059)
- **Test Weighted-F1**: 0.9377
- **Test Loss**: 0.2129

### Per-Class Metrics
| Class Name | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| healthy | 0.9408 | 0.9862 | 0.9630 | 145 |
| early_blight | 0.8936 | 0.8155 | 0.8528 | 103 |
| late_blight | 0.9333 | 0.9010 | 0.9169 | 202 |
| bacterial_spot | 0.9548 | 0.9462 | 0.9505 | 223 |
| leaf_mold | 0.8235 | 0.9899 | 0.8991 | 99 |
| septoria_leaf_spot | 0.9781 | 0.9040 | 0.9396 | 198 |
| spider_mites | 0.8779 | 0.9042 | 0.8909 | 167 |
| target_spot | 0.9070 | 0.8603 | 0.8830 | 136 |
| tomato_mosaic_virus | 0.8333 | 1.0000 | 0.9091 | 35 |
| tomato_yellow_leaf_curl_virus | 0.9839 | 0.9856 | 0.9848 | 557 |

### Top Confusion Pairs
- Actual target_spot -> Predicted spider_mites: **12** images
- Actual late_blight -> Predicted leaf_mold: **8** images
- Actual bacterial_spot -> Predicted tomato_yellow_leaf_curl_virus: **7** images
- Actual septoria_leaf_spot -> Predicted bacterial_spot: **7** images
- Actual early_blight -> Predicted late_blight: **6** images
- Actual early_blight -> Predicted leaf_mold: **6** images
- Actual spider_mites -> Predicted target_spot: **6** images
- Actual late_blight -> Predicted early_blight: **5** images
- Actual spider_mites -> Predicted healthy: **4** images
- Actual tomato_yellow_leaf_curl_virus -> Predicted spider_mites: **4** images

## Final Champion Checkpoints Summary
- **MODEL1**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model1\best_model1.pth
- **MODEL2**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model2_versions\model2_14class_refine_v1_f1_0.9335_acc_95.01.pth
- **MODEL3**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model3_versions\model3_potato_refine_v1_f1_0.9194_acc_93.67.pth
- **MODEL4**: C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model4_versions\model4_tomato_refine_v1_f1_0.9131_acc_93.47.pth