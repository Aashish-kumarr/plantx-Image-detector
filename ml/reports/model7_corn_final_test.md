# model7_corn Final Test Evaluation

- Checkpoint Path: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model7_corn_refine\best_refined_model.pth`
- Checkpoint Stage: `PREVIOUS_REFINE`
- Validation Accuracy: 90.62%
- Validation Macro-F1: 0.8841

## Test Metrics
- Test Samples: 388
- Test Loss: 0.3114
- Test Accuracy: 89.69%
- Correct: 348
- Incorrect: 40
- Macro Precision: 0.8671
- Macro Recall: 0.8802
- Macro-F1: 0.8700
- Weighted Precision: 0.9028
- Weighted Recall: 0.8969
- Weighted-F1: 0.8972

## Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.9832 | 1.0000 | 0.9915 | 117 |
| cercospora_leaf_spot_gray_leaf_spot | 0.6615 | 0.8269 | 0.7350 | 52 |
| common_rust | 0.9667 | 0.9667 | 0.9667 | 120 |
| northern_leaf_blight | 0.8571 | 0.7273 | 0.7869 | 99 |