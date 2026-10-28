# Models 5 & 7 Refinement Report

## Source Checkpoints
### model5_apple
- Source epoch   : 2
- Source Val Acc : 81.01%
- Source Macro-F1: 0.7583

### model7_corn
- Source epoch   : 5
- Source Val Acc : 80.73%
- Source Macro-F1: 0.762

## Refinement Results

### model5_apple
- Original Macro-F1 : 0.7583
- Best Refined F1   : 0.8907
- Delta             : +0.1325
- Best Refine Epoch : 4
- Stop Reason       : early_stopping
- Duration (s)      : 785.6
- Preferred Champion: **REFINED**

### model7_corn
- Original Macro-F1 : 0.7620
- Best Refined F1   : 0.8841
- Delta             : +0.1220
- Best Refine Epoch : 7
- Stop Reason       : max_epochs
- Duration (s)      : 1373.4
- Preferred Champion: **REFINED**

## Test Set Status
Test sets remain **LOCKED**. No test data was used during refinement.