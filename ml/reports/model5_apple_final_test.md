# model5_apple Final Test Evaluation

- Checkpoint Path: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model5_apple_final_refine\best_final_refined_model.pth`
- Checkpoint Stage: `FINAL_REFINE`
- Validation Accuracy: 92.41%
- Validation Macro-F1: 0.9135

## Test Metrics
- Test Samples: 319
- Test Loss: 0.2736
- Test Accuracy: 91.54%
- Correct: 292
- Incorrect: 27
- Macro Precision: 0.8994
- Macro Recall: 0.9124
- Macro-F1: 0.9053
- Weighted Precision: 0.9153
- Weighted Recall: 0.9154
- Weighted-F1: 0.9151

## Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.9394 | 0.9394 | 0.9394 | 165 |
| apple_scab | 0.8197 | 0.7937 | 0.8065 | 63 |
| black_rot | 0.9677 | 0.9524 | 0.9600 | 63 |
| cedar_apple_rust | 0.8710 | 0.9643 | 0.9153 | 28 |