# Models 5 & 7 Final Refinement Report

## Settings
- Starting LR : 1e-04
- Max epochs  : 5
- Early-stop patience: 3
- Scheduler   : ReduceLROnPlateau (max, factor=0.5, patience=1, min_lr=1e-5)
- AMP         : enabled

## Source Checkpoints
### model5_apple
- Source : `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model5_apple_refine\best_refined_model.pth`
- Source Val Acc  : 90.19%
- Source Macro-F1 : 0.8907

### model7_corn
- Source : `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\checkpoints\model7_corn_refine\best_refined_model.pth`
- Source Val Acc  : 90.62%
- Source Macro-F1 : 0.8841

## Refinement Results

### model5_apple
- Previous Macro-F1 : 0.8907
- Best Final F1     : 0.9135
- Delta             : +0.0227
- Best Epoch        : 4
- Stop Reason       : max_epochs
- Duration (s)      : 561.7
- **Preferred champion: FINAL-REFINED**

### model7_corn
- Previous Macro-F1 : 0.8841
- Best Final F1     : 0.8707
- Delta             : -0.0134
- Best Epoch        : 2
- Stop Reason       : early_stopping
- Duration (s)      : 872.7
- **Preferred champion: PREVIOUS-REFINED**

## Test Set Status
Test sets remain **LOCKED**. Zero test images were accessed during this stage.