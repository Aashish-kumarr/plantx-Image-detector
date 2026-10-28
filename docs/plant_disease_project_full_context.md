# DSP-Assisted Hierarchical Plant Disease Detection System
## Complete Project Context / Handoff Document

> **Purpose of this document:**  
> This file is a complete context handoff for a new chatbot, developer, frontend engineer, backend engineer, or ML engineer who needs to understand the project without reading the full prior conversation.
>
> It intentionally includes the project concept, current architecture, dataset decisions, DSP preprocessing, data management, Model 1 preparation, training strategy, testing strategy, backend/frontend implications, current file structure, known decisions, completed steps, pending steps, and how the implementation differs from the original synopsis.
>
> **Important:** The project implementation evolved beyond the original synopsis. The current implementation should be treated as the source of truth for development unless the team explicitly decides to revert.

---

# 1. Project Identity

## Project title from the original synopsis

**DSP-ASSISTED EXPLAINABLE AND LIGHTWEIGHT PLANT DISEASE DETECTION SYSTEM**

Original academic context:

- Digital Signal Processing Laboratory project
- Domain:
  - Digital Image Processing
  - Deep Learning
  - Explainable AI
  - Edge AI
- Academic Year: 2026

The original synopsis was written around a 6-class potato/tomato disease classifier using classical DSP preprocessing plus CNNs.

The implementation has since evolved into a **hierarchical 4-model system**.

---

# 2. Current Project Goal

The current project is intended to accept an image uploaded by a user and determine:

1. whether the image is a plant leaf or not,
2. which supported crop the leaf belongs to,
3. which disease/health condition the crop has,
4. reject unsupported crops rather than forcing a disease prediction.

The current supported disease targets are:

### Potato
- Healthy
- Early Blight
- Late Blight

### Tomato
- Healthy
- Early Blight
- Late Blight

The long-term user-facing product will contain:

- image upload,
- preprocessing,
- hierarchical ML inference,
- final crop/disease result,
- model score/confidence-like score,
- error/rejection states,
- future explainability output,
- frontend UI,
- backend API,
- possible local/offline deployment later.

---

# 3. Current Final Inference Architecture

The current model architecture is hierarchical.

```text
User uploads image
        |
        v
DSP preprocessing
        |
        v
MODEL 1
Leaf vs Non-Leaf
        |
        |-- Non-Leaf --> Reject:
        |               "No suitable plant leaf detected"
        |
        v
MODEL 2
Potato / Tomato / Other Plant
        |
        |-- Other Plant --> Reject:
        |                  "Unsupported crop"
        |
        |-- Potato
        |      |
        |      v
        |   MODEL 3
        |   Potato disease classifier
        |   - Healthy
        |   - Early Blight
        |   - Late Blight
        |
        |-- Tomato
               |
               v
            MODEL 4
            Tomato disease classifier
            - Healthy
            - Early Blight
            - Late Blight
```

---

# 4. Model Responsibilities

## Model 1 — Leaf vs Non-Leaf

### Purpose

Determine whether the uploaded image should enter the crop/disease pipeline.

### Labels

```text
0 = non_leaf
1 = leaf
```

### Current training sources

Leaf:
- PlantVillage `color` dataset

Non-leaf:
- filtered Caltech-256 subset
- filtered Places365 validation subset

### Important scope decision

For the first version, difficult edge cases are intentionally postponed.

Examples not being specially engineered for yet:

- cactus
- artificial leaves
- flower-only images
- strange plant structures
- distant vegetation
- leaf-like objects

The current priority is a strong baseline Leaf vs Non-Leaf classifier.

---

## Model 2 — Crop Classifier

### Planned labels

```text
0/label mapping to be finalized
Potato
Tomato
Other Plant
```

### Purpose

Prevent the system from forcing every valid leaf into potato or tomato.

### Planned data source

PlantVillage.

Mapping concept:

```text
All Potato classes -> Potato
All Tomato classes -> Tomato
All remaining PlantVillage crop classes -> Other Plant
```

The disease label is ignored for Model 2.

Example:

```text
Apple___healthy -> Other Plant
Corn___Common_rust -> Other Plant
Potato___Late_blight -> Potato
Tomato___Early_blight -> Tomato
```

---

## Model 3 — Potato Disease Classifier

### Labels

- Healthy
- Early Blight
- Late Blight

### PlantVillage counts

- Potato Early Blight: 1000
- Potato Late Blight: 1000
- Potato Healthy: 152

This model has a major class-imbalance issue because Healthy is much smaller than the two disease classes.

Expected mitigation later:

- training-only augmentation,
- class-weighted loss,
- possibly balanced sampling,
- untouched validation/test splits.

---

## Model 4 — Tomato Disease Classifier

### Labels

- Healthy
- Early Blight
- Late Blight

### PlantVillage counts

- Tomato Early Blight: 1000
- Tomato Late Blight: 1909
- Tomato Healthy: 1591

This dataset is considerably better balanced than the potato subset.

---

# 5. Original Synopsis vs Current Implementation

The original synopsis proposed:

```text
Raw RGB branch
        +
DSP RGB branch
        |
ResNet-9
and
MobileNetV2
        |
Single 6-class output:
Potato Healthy
Potato Early
Potato Late
Tomato Healthy
Tomato Early
Tomato Late
```

The synopsis also explicitly proposed:

- Raw RGB vs DSP comparison,
- ResNet-9 reference model,
- MobileNetV2 lightweight model,
- Grad-CAM and/or SHAP,
- INT8 evaluation,
- field-style external validation.

The **current implementation differs** in important ways:

1. The system is now hierarchical with 4 models.
2. Model 1 is Leaf vs Non-Leaf.
3. Model 2 is Potato/Tomato/Other Plant.
4. Models 3 and 4 are crop-specific disease classifiers.
5. The current decision is to train models using **DSP-preprocessed input only**.
6. The raw-RGB training branch has been removed from the active implementation.
7. ResNet-9 is still retained and therefore remains aligned with the original synopsis.
8. MobileNetV2 has not yet been implemented.
9. Grad-CAM/SHAP has not yet been implemented.
10. INT8 has not yet been implemented.

Therefore, if the final submitted report/synopsis must exactly match implementation, the synopsis will eventually need revision.

---

# 6. Why This Is Still a DSP Project

The project treats each color image as a sampled and quantized 2D signal.

The classical DSP / image processing pipeline includes:

- RGB to HSV conversion,
- Gaussian filtering,
- histogram-based thresholding,
- Otsu automatic thresholding,
- binary masks,
- morphological opening,
- morphological closing,
- connected component filtering,
- foreground isolation,
- preservation of original RGB foreground values.

The CNN then performs learned convolutional filtering after the classical DSP stage.

---

# 7. DSP Preprocessing Pipeline — Current Model 1 Implementation

The current Model 1 DSP preprocessing pipeline is finalized for the first training version.

## Pipeline

```text
Original image
    |
    v
Load image
    |
    v
Force RGB
    |
    v
RGB -> HSV
    |
    v
Use S (Saturation) channel
    |
    v
Gaussian Blur
kernel = 5x5
    |
    v
Otsu automatic thresholding
    |
    v
Binary foreground mask
    |
    v
Morphological Opening
    |
    v
Morphological Closing
ellipse kernel = 5x5
2 iterations
    |
    v
Connected Components
    |
    v
Keep largest isolated blob
    |
    v
Apply mask to ORIGINAL RGB image
    |
    v
Background -> Black
Foreground -> original RGB values preserved
    |
    v
Resize while preserving aspect ratio
    |
    v
Center crop
    |
    v
224 x 224 RGB DSP image
    |
    v
Training augmentation (train only)
    |
    v
ToTensor
    |
    v
DSP-specific normalization
    |
    v
CNN
```

---

# 8. Important DSP Design Rule

The final CNN input is **not grayscale**.

HSV and binary representations are only intermediate signal-processing tools.

The final network receives:

```text
3-channel RGB image
with foreground retained
and background masked to black
```

This preserves useful lesion color information.

---

# 9. DSP Mask Validation

The preprocessing was visually tested on:

- 20 PlantVillage images
- 10 Caltech-256 images
- 10 Places365 images

Saved inspection directory:

```text
reports/model1_dsp_samples/
```

For each selected sample:

- original image,
- binary mask,
- DSP-processed final RGB image

were saved.

---

# 10. Large-Scale DSP Mask Audit

A larger audit was conducted on training images only.

Sample:

- 1000 PlantVillage
- 1000 Caltech-256
- 1000 Places365

## Results

### PlantVillage Leaf

- Mean foreground: 36.32%
- Median: 37.80%
- 25th percentile: 24.86%
- 75th percentile: 47.46%

### Caltech-256 Non-Leaf

- Mean foreground: 26.55%
- Median: 20.94%
- 25th percentile: 8.97%
- 75th percentile: 39.70%

### Places365 Non-Leaf

- Mean foreground: 25.41%
- Median: 21.33%
- 25th percentile: 10.21%
- 75th percentile: 36.39%

### Combined Non-Leaf

- Mean foreground: 25.98%
- Median: 21.27%
- 25th percentile: 9.56%
- 75th percentile: 37.74%

### Important diagnostics

- Leaf images with foreground <1%: 0.40%
- Non-leaf images with foreground <1%: 3.55%
- Non-leaf images with foreground >25%: 43.60%

### Interpretation

The DSP pipeline does **not** create an obvious trivial shortcut such as:

```text
Leaf -> large colored area
Non-leaf -> black image
```

There is substantial overlap between foreground ratios.

This is desirable because Model 1 must learn visual structure/texture rather than a simple black-background ratio.

Audit files:

```text
reports/model1_dsp_large_audit.txt
reports/model1_dsp_mask_statistics.csv
```

---

# 11. Dataset Root

Current local project location:

```text
C:\Users\ak021\Desktop\workspace\plantDx\plant image detector
```

Dataset directory:

```text
C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset
```

---

# 12. PlantVillage Dataset

Exact inspected path:

```text
C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\plantvillage dataset
```

The local PlantVillage copy contains:

- color
- grayscale
- segmented

versions.

## Total physical images across all versions

```text
162,916
```

## Important

The three versions correspond largely to representations of the same underlying images.

For Model 1, **only the `color` copy is used**.

The grayscale and segmented versions are not included.

This avoids:

- duplicate semantic samples,
- representation leakage,
- train/test leakage through alternate copies of the same source image.

## Model 1 PlantVillage color count

```text
54,305
```

All are labeled:

```text
leaf = 1
source = plantvillage
```

---

# 13. PlantVillage Classes

38 distinct crop/disease classes were discovered.

## Apple

- Apple scab: 630
- Black rot: 621
- Cedar apple rust: 275
- Healthy: 1645

## Blueberry

- Healthy: 1502

## Cherry

- Powdery mildew: 1052
- Healthy: 854

## Corn / Maize

- Cercospora leaf spot / Gray leaf spot: 513
- Common rust: 1192
- Northern Leaf Blight: 985
- Healthy: 1162

## Grape

- Black rot: 1180
- Esca / Black Measles: approximately 1383
- Leaf blight / Isariopsis Leaf Spot: 1076
- Healthy: 423

## Orange

- Huanglongbing / Citrus greening: 5507

## Peach

- Bacterial spot: 2297
- Healthy: 360

## Pepper, bell

- Bacterial spot: 997
- Healthy: 1478

## Potato

- Early blight: 1000
- Late blight: 1000
- Healthy: 152

## Raspberry

- Healthy: 371

## Soybean

- Healthy: 5090

## Squash

- Powdery mildew: 1835

## Strawberry

- Leaf scorch: 1109
- Healthy: 456

## Tomato

- Bacterial spot: 2127
- Early blight: 1000
- Late blight: 1909
- Leaf Mold: 952
- Septoria leaf spot: 1771
- Spider mites / Two-spotted spider mite: 1676
- Target Spot: 1404
- Tomato Yellow Leaf Curl Virus: 5357
- Tomato mosaic virus: 373
- Healthy: 1591

---

# 14. PlantVillage Image Properties

Inspected images:

- 256 x 256
- RGB
- file types:
  - .jpg
  - .jpeg
  - .png

No unreadable/corrupted images were found in the dataset inspection.

---

# 15. Non-Leaf Dataset Strategy

The team originally considered:

- Caltech101,
- Caltech256,
- ImageNet,
- Places365.

Caltech101 alone was considered too small.

ImageNet was avoided because downloading the full training archive would be unnecessarily large for the available local storage and the project only needs a non-leaf negative set.

The final initial solution combines:

```text
Caltech-256 SAFE subset
+
Places365 filtered validation subset
```

---

# 16. Caltech-256 Dataset

Exact path:

```text
C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\caltech256
```

Source:

Caltech official data repository.

Original dataset:

- 30,607 images
- 257 class folders
  - 256 object categories
  - 1 clutter category
- .jpg

A category audit was performed.

## SAFE_NON_LEAF

- 240 categories
- 27,882 images

## EXCLUDE_OR_REVIEW

- 17 categories
- 2,725 images

Excluded categories included botanical or ambiguous categories such as:

- bonsai
- cactus
- fern
- palm tree
- grapes
- tomato
- watermelon
- some foliage-associated insect/animal categories
- clutter

The exact audit report:

```text
reports/caltech256_category_audit.txt
```

Only SAFE_NON_LEAF categories are used for Model 1.

---

# 17. Places365 Non-Leaf Dataset

Exact path:

```text
C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\places365_non_leaf
```

Source:

Official Places365 256x256 validation set.

A filtered subset was created.

Selection:

- 289 categories
- 100 images per category
- total = 28,900 images

76 strongly natural/botanical categories were excluded.

Examples of categories intended to be avoided:

- forest
- garden
- farm
- field
- vegetation
- trees
- plants
- flowers
- orchard
- greenhouse
- botanical garden
- grass-dominated environments

The Places365 subset preserves original labels.

---

# 18. Model 1 Dataset Composition

Current Model 1 master dataset:

```text
Total images = 111,087
```

## Leaf

PlantVillage color:

```text
54,305
48.89%
```

## Non-Leaf

Caltech-256 SAFE:

```text
27,882
```

Places365 filtered:

```text
28,900
```

Combined Non-Leaf:

```text
56,782
51.11%
```

This is close to balanced.

No further non-leaf dataset is currently required.

---

# 19. Model 1 Master Metadata

File:

```text
metadata/model1_master.csv
```

Columns:

```text
image_path
label
label_id
source
original_class
```

Example concept:

```text
...\PlantVillage\color\Tomato___healthy\image.jpg,leaf,1,plantvillage,Tomato___healthy
...\caltech256\009.bear\image.jpg,non_leaf,0,caltech256,009.bear
...\places365_non_leaf\airport_terminal\image.jpg,non_leaf,0,places365,airport_terminal
```

## Label mapping

```text
non_leaf = 0
leaf = 1
```

---

# 20. Metadata Validation

Validation results:

- total images: 111,087
- missing files: 0
- unreadable/corrupted: 0
- duplicate paths: 0
- forbidden PlantVillage versions: 0
- excluded Caltech categories included: 0
- exact duplicate image-content matches: 37

The 37 exact-content duplicates were **not deleted**.

Instead, the split logic groups duplicates so identical content cannot occur across multiple splits.

---

# 21. Model 1 Train / Validation / Test Split

File:

```text
metadata/model1_split.csv
```

Split:

```text
80% train
10% validation
10% test
```

Random seed:

```text
42
```

Stratification uses:

```text
(label, source)
```

This preserves:

- Leaf/Non-Leaf proportions
- Caltech representation
- Places representation
- PlantVillage representation

## Counts

### Train

```text
88,869
```

- Non-Leaf: 45,425
- Leaf: 43,444

Sources:

- PlantVillage: 43,444
- Caltech-256: 22,305
- Places365: 23,120

### Validation

```text
11,108
```

- Non-Leaf: 5,678
- Leaf: 5,430

Sources:

- PlantVillage: 5,430
- Caltech-256: 2,788
- Places365: 2,890

### Test

```text
11,110
```

- Non-Leaf: 5,679
- Leaf: 5,431

Sources:

- PlantVillage: 5,431
- Caltech-256: 2,789
- Places365: 2,890

---

# 22. Duplicate Leakage Protection

Exact duplicate images were identified using SHA-256 hashes.

Duplicate-content groups:

```text
37
```

Duplicate groups crossing train/val/test:

```text
0
```

Rule:

```text
All identical SHA-256 content must remain in the same split.
```

This protects against inflated validation/test performance due to exact duplicate leakage.

---

# 23. Basic Deterministic Image Pipeline

An earlier basic preprocessing step was completed before DSP-specific normalization was finalized.

The deterministic final pipeline for Model 1 is now:

```text
DSPTransform
    |
ToTensor
    |
Normalize with DSP-specific mean/std
```

Final tensor:

```text
shape = 3 x 224 x 224
dtype = torch.float32
```

No NaNs or Infs were found.

---

# 24. DSP-Specific Normalization

Normalization statistics were recomputed **after DSP preprocessing** using **training images only**.

This is important because the earlier pre-DSP statistics are obsolete.

Normalization is stored in:

```text
metadata/model1_normalization.json
```

Expected contents include:

- mean
- std
- image_size
- computed_from
- number_of_images
- dsp_preprocessing = true
- split = train
- DSP configuration

**Important handoff note:**  
The exact numerical RGB mean/std values were not included in the chatbot conversation summary supplied to this document. They exist locally in `metadata/model1_normalization.json`. Backend/training code must load those values from that file rather than inventing or replacing them with ImageNet defaults.

Do not use ImageNet mean/std for this model unless explicitly retraining with a different design.

---

# 25. Training-Only Augmentation

Training augmentation has been implemented and validated.

Pipeline:

```text
Original
-> DSP
-> 224 x 224 RGB
-> training augmentation
-> ToTensor
-> DSP normalization
```

Validation/test/inference:

```text
Original
-> DSP
-> 224 x 224 RGB
-> ToTensor
-> DSP normalization
```

No random augmentation is allowed in validation/test/inference.

The augmentation design is intentionally conservative.

Implemented/requested augmentation family:

- RandomHorizontalFlip, p approximately 0.5
- rotation approximately +/-15 degrees
- mild affine translation
- mild scale variation
- mild brightness variation
- mild contrast variation

Intentionally avoided at this stage:

- MixUp
- CutMix
- Random Erasing
- heavy perspective distortion
- extreme rotations
- grayscale conversion
- strong hue changes
- strong saturation changes
- artificial noise

Reason:

Disease classification later may rely on realistic lesion color.

Visual samples:

```text
reports/model1_augmentation_samples/
```

Validation showed:

- eval transform deterministic
- train transform variable
- output shape correct
- no NaNs/Infs

---

# 26. PyTorch Dataset / DataLoader

File:

```text
src/datasets/model1_dataset.py
```

Dataset:

```text
Model1Dataset
```

It reads:

```text
metadata/model1_split.csv
```

Returns:

```python
(image_tensor, label)
```

Expected image:

```text
3 x 224 x 224
torch.float32
```

Expected label:

```text
0 or 1
```

---

# 27. DataLoader Validation

Verified:

- Train = 88,869
- Val = 11,108
- Test = 11,110

Batch size:

```text
64
```

Batch shape:

```text
[64, 3, 224, 224]
```

Labels:

```text
[64]
```

No NaNs/Infs.

---

# 28. Initial DataLoader Performance

An early sanity check measured about:

```text
50 images/sec
```

with heavy CPU DSP preprocessing.

This initially suggested approximately 30 minutes of input processing per epoch.

However, later optimized benchmarking showed the earlier configuration was highly inefficient.

---

# 29. Cache Benchmark

Because DSP was initially CPU-bound, a cache benchmark was performed.

1000 DSP-processed images were tested.

## Lossless WebP

- ~19.5 KB/image
- projected full cache ~2.07 GB

## PNG

- ~29.6 KB/image
- projected full cache ~3.14 GB

Both preserved exact RGB pixels.

But cached WebP loading was slower:

```text
~21.5 images/sec
```

whereas on-the-fly DSP at that time was:

```text
~49.7 images/sec
```

Conclusion:

**Do not cache DSP output using PNG/WebP for training.**

The final project uses on-the-fly DSP preprocessing.

---

# 30. DataLoader Optimization

The major performance issue turned out to be OpenCV threading oversubscription.

## Original naive setup

```text
4 PyTorch workers
OpenCV automatic internal threading
```

Measured later:

```text
~140.8 images/sec
```

## Final optimized setup

```text
num_workers = 8
persistent_workers = True
prefetch_factor = 4
pin_memory = True when CUDA available
OpenCV threads = 1 per worker
```

Critical setting:

```python
cv2.setNumThreads(1)
```

Reason:

Without this, every PyTorch worker could independently spawn many OpenCV CPU threads, causing severe CPU contention.

## Optimized throughput

```text
~720.8 images/sec
```

Estimated pure pipeline time for one 88,869-image training epoch:

```text
~123 seconds
~2 minutes
```

The optimized parameters were hardcoded as defaults in:

```text
src/datasets/model1_dataset.py
```

Do not remove this optimization casually.

---

# 31. Local Hardware / Development Environment

Primary training machine:

- Windows laptop
- Lenovo LOQ
- Intel i7 14th-generation class CPU
- NVIDIA RTX 5060
- local SSD
- PyTorch
- CUDA
- VS Code

The user explicitly does **not** want Jupyter for this project.

Development/training environment:

```text
VS Code
Python
PyTorch
CUDA
OpenCV
NumPy
Matplotlib
scikit-learn
```

Potential later components:

- Grad-CAM
- SHAP
- ONNX
- TFLite or other mobile runtime
- FastAPI
- React frontend

---

# 32. Training Philosophy

All neural networks are intended to be trained locally.

Important constraints:

- no cloud training required,
- no pretrained model weights,
- no transfer learning,
- random initialization,
- local RTX 5060 CUDA training.

“From scratch” means:

- known architecture is allowed,
- weights start random,
- training is done on project data,
- no ImageNet pretrained weights are loaded.

---

# 33. Model 1 Architecture Decision

Model 1 will use:

```text
ResNet-9 style architecture
```

trained from scratch.

This remains consistent with the original project's ResNet-9 reference direction.

Planned file:

```text
src/models/model1_resnet9.py
```

---

# 34. Planned Model 1 ResNet-9

Input:

```text
3 x 224 x 224
```

Output:

```text
2 raw logits
```

No Sigmoid or Softmax inside the network.

Loss will later use raw logits with CrossEntropyLoss.

Architecture:

```text
Input
 |
Conv 3 -> 64, 3x3
BatchNorm
ReLU
 |
Conv 64 -> 128, 3x3
BatchNorm
ReLU
MaxPool
 |
Residual Block
  Conv 128 -> 128
  BN
  ReLU
  Conv 128 -> 128
  BN
  Skip Add
  ReLU
 |
Conv 128 -> 256
BN
ReLU
MaxPool
 |
Conv 256 -> 512
BN
ReLU
MaxPool
 |
Residual Block
  Conv 512 -> 512
  BN
  ReLU
  Conv 512 -> 512
  BN
  Skip Add
  ReLU
 |
AdaptiveAvgPool2d(1)
 |
Flatten
 |
Linear 512 -> 2
 |
Raw logits
```

This corresponds to eight convolution layers plus final classifier.

Weight initialization:

- Kaiming / He for Conv layers
- standard sensible BatchNorm initialization
- sensible Linear initialization
- no checkpoint loaded
- no internet weights

---

# 35. Step 15 Status

At the time this handoff document was generated:

**Step 15 had been prepared but not yet reported as completed.**

Step 15 is intended to:

- implement ResNet-9,
- validate dummy forward pass,
- validate real batch on RTX 5060,
- report parameter count,
- report FP32 size,
- report CUDA memory,
- run one gradient sanity check,
- NOT train for an epoch.

Expected file:

```text
reports/model1_resnet9_architecture.txt
```

A future chatbot must not assume Step 15 succeeded unless this report exists.

---

# 36. Planned Training Configuration — Not Yet Locked

The following had been discussed as likely choices but were intentionally not yet finalized in the step-by-step workflow:

- CrossEntropyLoss
- AdamW
- learning rate around 1e-3 as an initial idea
- scheduler
- mixed precision
- maximum around 30 epochs initially
- early stopping
- best and last checkpoints

These must be finalized only after architecture validation and a small benchmark.

Do not assume final hyperparameters until the local training setup report confirms them.

---

# 37. Planned Training Workflow for Model 1

After Step 15:

1. define training machinery,
2. define loss,
3. define optimizer,
4. define scheduler,
5. enable mixed precision,
6. implement checkpointing,
7. implement metrics,
8. run CUDA sanity test,
9. run 1-epoch benchmark,
10. inspect results,
11. run full training,
12. evaluate on untouched test set,
13. inspect confusion matrix,
14. inspect false positives,
15. inspect false negatives,
16. save final model,
17. build Model 1 inference wrapper.

---

# 38. Intended Model 1 Metrics

Model 1 should not be judged only by accuracy.

Required metrics:

- Accuracy
- Precision
- Recall
- F1
- confusion matrix

Important error categories:

```text
Leaf -> predicted Non-Leaf
Non-Leaf -> predicted Leaf
```

For a real product, false rejection of a valid leaf and false acceptance of unrelated images both matter.

---

# 39. Model 1 Inference Behavior

Planned runtime logic:

```python
result = model1(processed_image)

if result == "non_leaf":
    return {
        "accepted": False,
        "reason": "no_leaf_detected"
    }

return {
    "accepted": True
}
```

Suggested user-facing text:

```text
No suitable plant leaf detected.
Please upload a clear image of a plant leaf.
```

Do not expose raw technical class IDs to end users.

---

# 40. Model 2 Data Strategy

After Model 1 is fully trained and evaluated, Model 2 should be created.

PlantVillage mapping:

```text
Potato_* -> Potato
Tomato_* -> Tomato
Everything else -> Other Plant
```

Potential challenge:

`Other Plant` contains many more samples than Potato.

Possible solutions:

- controlled sampling from Other Plant,
- class-weighted loss,
- balanced sampling,
- augmentation,
- preserve val/test naturally.

The exact strategy is not yet finalized.

---

# 41. Model 2 Runtime Behavior

```text
Model 2 -> Potato
    -> Model 3

Model 2 -> Tomato
    -> Model 4

Model 2 -> Other Plant
    -> Reject as unsupported
```

Suggested user-facing response:

```text
A plant leaf was detected, but this crop is not currently supported.
Currently supported crops: Potato and Tomato.
```

---

# 42. Model 3 Training Considerations

Potato data is highly imbalanced.

Counts:

```text
Healthy = 152
Early Blight = 1000
Late Blight = 1000
```

Important:

Do not balance validation/test via synthetic duplication.

Potential training-only techniques:

- augmentation of minority class,
- class weights,
- balanced sampler.

Model output:

```text
3 logits
```

No Softmax inside the network if using CrossEntropyLoss.

---

# 43. Model 4 Training Considerations

Tomato target subset:

```text
Healthy = 1591
Early Blight = 1000
Late Blight = 1909
```

This is significantly more balanced.

Model output:

```text
3 logits
```

---

# 44. DSP Pipeline Across Models

The project direction is to keep DSP preprocessing central.

Unless experiments later show a specific reason to alter it, the architecture concept is:

```text
Uploaded original image
       |
       v
Shared DSP processing
       |
       +-> Model 1
       |
       +-> Model 2
       |
       +-> Model 3 or Model 4
```

A backend implementation should preferably avoid needlessly recomputing identical deterministic DSP work several times for the same request.

Potential efficient runtime design:

```text
Upload image
   |
Run DSP once
   |
Produce processed 224x224 RGB representation
   |
Reuse representation for all relevant model stages
```

However, this is only valid if Models 2–4 are ultimately trained using exactly the same preprocessing dimensions/statistics expectations.

If each model later receives model-specific normalization statistics, the masked 224x224 RGB image may be reusable while tensor normalization is applied separately per model.

---

# 45. Backend Architecture Recommendation

The backend should be designed independently of the training scripts.

Suggested stack:

```text
FastAPI
Python
PyTorch
OpenCV
Pillow
NumPy
```

Why FastAPI:

- same language as ML pipeline,
- easy model loading,
- easy OpenCV/PyTorch integration,
- straightforward multipart file upload,
- automatic OpenAPI docs.

Possible backend structure:

```text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── inference_service.py
│   │   ├── dsp_service.py
│   │   └── model_loader.py
│   ├── models/
│   │   └── model_registry.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   └── utils/
│       └── image_validation.py
├── checkpoints/
├── requirements.txt
└── README.md
```

Do not duplicate model architecture/preprocessing definitions unnecessarily.

Prefer importing shared inference-safe modules from the ML project.

---

# 46. Backend Model Registry

Models should load once when the server starts, not once per request.

Concept:

```python
model1 = load_model(...)
model2 = load_model(...)
model3 = load_model(...)
model4 = load_model(...)

model1.eval()
model2.eval()
model3.eval()
model4.eval()
```

Use:

```python
torch.inference_mode()
```

during prediction.

Backend should expose a health status indicating whether required checkpoints loaded successfully.

---

# 47. Backend Runtime Pipeline

Suggested high-level function:

```python
def predict_plant_disease(image):
    validate_image(image)

    dsp_image = preprocess_dsp(image)

    leaf_result = run_model1(dsp_image)

    if leaf_result.label == "non_leaf":
        return rejection_response(...)

    crop_result = run_model2(dsp_image)

    if crop_result.label == "other":
        return unsupported_crop_response(...)

    if crop_result.label == "potato":
        disease_result = run_model3(dsp_image)

    elif crop_result.label == "tomato":
        disease_result = run_model4(dsp_image)

    return final_response(...)
```

---

# 48. Suggested Prediction API

Endpoint:

```text
POST /api/v1/predict
```

Content type:

```text
multipart/form-data
```

Field:

```text
image
```

Potential response for success:

```json
{
  "success": true,
  "leaf_detected": true,
  "crop": "tomato",
  "disease": "early_blight",
  "display_name": "Tomato - Early Blight",
  "scores": {
    "leaf": 0.998,
    "crop": 0.973,
    "disease": 0.941
  },
  "supported": true,
  "explanation_available": false
}
```

Do not describe scores as medically/biologically guaranteed probabilities unless proper calibration is later implemented.

Prefer wording:

```text
model score
```

or:

```text
prediction score
```

---

# 49. Non-Leaf API Response

Example:

```json
{
  "success": true,
  "leaf_detected": false,
  "supported": false,
  "reason": "no_leaf_detected",
  "message": "No suitable plant leaf detected. Please upload a clear image of a plant leaf."
}
```

HTTP request itself succeeded, so a 200 response can be reasonable.

The prediction result is a rejection, not a server error.

---

# 50. Unsupported Crop API Response

Example:

```json
{
  "success": true,
  "leaf_detected": true,
  "supported": false,
  "reason": "unsupported_crop",
  "crop": "other",
  "message": "This plant is not currently supported. Please upload a potato or tomato leaf."
}
```

---

# 51. Invalid Upload API Response

Examples:

- unsupported extension,
- file is not an image,
- decoding fails,
- empty file,
- file too large.

Suggested:

```json
{
  "success": false,
  "error": "invalid_image",
  "message": "The uploaded file could not be processed as an image."
}
```

Use a 4xx status for invalid user input.

---

# 52. Backend Image Validation

Recommended checks:

- maximum upload size,
- MIME type,
- actual image decoding,
- RGB conversion,
- pixel dimensions,
- corrupted image handling,
- EXIF orientation normalization if required,
- no trust in filename extension alone.

Do not save user uploads permanently unless a product requirement explicitly asks for history.

---

# 53. GPU / CPU Backend Behavior

Development environment has RTX 5060.

Backend should support:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

However:

- training validation explicitly required CUDA,
- production inference may optionally support CPU,
- frontend/backend should not assume every deployment has RTX hardware.

If CPU inference is too slow, deployment requirements should state GPU support.

---

# 54. Frontend Recommendation

Suggested stack:

```text
React
Vite
Tailwind CSS
Axios or fetch
```

A Next.js frontend is also possible, but a standard React/Vite frontend is sufficient because the heavy logic resides in FastAPI.

---

# 55. Suggested Frontend Pages

Minimum first version:

```text
/
Home / Upload
```

Optional:

```text
/about
/how-it-works
/supported-diseases
```

Primary goal:

Make the upload/prediction workflow extremely clear.

---

# 56. Main Frontend User Flow

```text
Landing page
     |
Upload / drag-drop image
     |
Image preview
     |
Analyze button
     |
Loading state
     |
Backend prediction
     |
Result card
```

Possible result states:

1. Non-leaf rejection
2. Unsupported crop
3. Potato Healthy
4. Potato Early Blight
5. Potato Late Blight
6. Tomato Healthy
7. Tomato Early Blight
8. Tomato Late Blight
9. Invalid upload
10. Backend/model error

---

# 57. Frontend Upload Component

Should support:

- drag and drop,
- file picker,
- image preview,
- replace image,
- remove image,
- Analyze button,
- loading state,
- validation messages.

Recommended accepted formats:

- JPEG
- PNG
- WebP if backend supports it reliably

The frontend should validate file size early, but backend validation remains authoritative.

---

# 58. Frontend Prediction Loading State

DSP and multi-model inference may take measurable time.

Use a clear loading message such as:

```text
Analyzing leaf image...
```

Do not display fake progress percentages unless the backend supplies real progress.

A subtle staged message is possible:

```text
Checking image...
Identifying crop...
Analyzing disease...
```

but only if implementation does not falsely claim specific completed stages.

---

# 59. Frontend Result Design

Success result should prominently show:

```text
Tomato
Early Blight
```

Possible secondary information:

- prediction score,
- leaf detection score,
- crop score,
- disease score,
- supported crop status,
- future Grad-CAM visualization,
- short disease description,
- recommendation disclaimer.

Do not overstate model certainty.

---

# 60. Explainability — Future/Planned

Original synopsis requires:

- Grad-CAM and/or SHAP

Purpose:

- visualize image regions influencing prediction,
- inspect whether the disease model attends to lesion/leaf tissue,
- reduce reliance on irrelevant background.

Potential API later:

```json
{
  "explanation_available": true,
  "heatmap_url": "/..."
}
```

or return base64/image endpoint.

This is not implemented yet.

---

# 61. Mobile / Edge — Future/Planned

Original project includes:

- lightweight architecture comparison,
- MobileNetV2,
- INT8 evaluation,
- potential offline smartphone deployment.

Potential future path:

```text
PyTorch checkpoint
-> ONNX
-> mobile-compatible runtime / TFLite / ONNX Runtime Mobile
-> quantization
```

Do not implement this before the base four-model pipeline is stable.

---

# 62. External / Field Validation — Future

PlantVillage is a controlled dataset.

The original project explicitly recognizes that PlantVillage accuracy alone does not prove field reliability.

Potential external source:

- PlantDoc
- separately collected field images

Evaluation should distinguish:

```text
Controlled test performance
vs
Field-style performance
```

Do not mix these results into a single misleading number.

---

# 63. Important Testing Principle

The test set must remain untouched until final model selection.

Do not:

- tune on test,
- inspect test labels while changing hyperparameters,
- augment test,
- rebalance test,
- use test data to compute normalization,
- use test to choose thresholds.

Use:

```text
Train -> fit weights
Validation -> select/tune
Test -> final report
```

---

# 64. Data Leakage Protections Already Implemented

- PlantVillage color only
- no grayscale/segmented replicas
- duplicate SHA-256 grouping
- normalization computed from train only
- augmentation train only
- val/test deterministic
- train/val/test mutually exclusive
- source-stratified split

These rules should be retained for Models 2–4.

---

# 65. Reproducibility

Important reproducibility settings:

```text
split seed = 42
```

Future training should save:

- random seeds,
- configuration,
- learning rate,
- batch size,
- optimizer,
- scheduler,
- augmentation parameters,
- normalization stats,
- model architecture,
- checkpoint epoch,
- validation metrics.

---

# 66. Suggested Checkpoint Layout

```text
checkpoints/
├── model1/
│   ├── best_model1.pth
│   └── last_model1.pth
├── model2/
│   ├── best_model2.pth
│   └── last_model2.pth
├── model3/
│   ├── best_model3.pth
│   └── last_model3.pth
└── model4/
    ├── best_model4.pth
    └── last_model4.pth
```

Actual filenames can change, but backend configuration should not hardcode paths scattered across source files.

Use a central config/model registry.

---

# 67. Current / Suggested Project Structure

Current concepts already use files such as:

```text
plant image detector/
├── dataset/
│   ├── plantvillage dataset/
│   ├── caltech256/
│   └── places365_non_leaf/
│
├── metadata/
│   ├── model1_master.csv
│   ├── model1_split.csv
│   └── model1_normalization.json
│
├── reports/
│   ├── caltech256_category_audit.txt
│   ├── model1_metadata_report.txt
│   ├── model1_split_report.txt
│   ├── model1_preprocessing_report.txt
│   ├── model1_dsp_preprocessing_report.txt
│   ├── model1_dsp_large_audit.txt
│   ├── model1_dsp_mask_statistics.csv
│   ├── model1_dsp_normalization_report.txt
│   ├── model1_augmentation_report.txt
│   ├── model1_dataloader_report.txt
│   ├── model1_dsp_cache_benchmark.txt
│   ├── model1_dataloader_optimization.txt
│   ├── model1_dsp_samples/
│   └── model1_augmentation_samples/
│
├── src/
│   ├── datasets/
│   │   ├── model1_dsp_preprocessing.py
│   │   ├── model1_transforms.py
│   │   └── model1_dataset.py
│   │
│   └── models/
│       └── model1_resnet9.py    # planned/Step 15
│
├── checkpoints/
├── results/
├── train.py
├── evaluate.py
├── inference.py
├── config.py
└── utils.py
```

Not every top-level file above is guaranteed to exist yet; this represents the intended organization plus files explicitly created during the workflow.

---

# 68. Recommended Separation Between ML and Web Application

Do not put frontend/backend logic inside training scripts.

Recommended repository layout later:

```text
project/
├── ml/
│   ├── src/
│   ├── metadata/
│   ├── checkpoints/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── backend/
│   └── FastAPI application
│
├── frontend/
│   └── React/Vite application
│
└── README.md
```

Shared inference-safe ML code can be imported by backend.

---

# 69. Training Code Must Not Leak Into API Concerns

Training-only concepts:

- random augmentation,
- optimizer,
- scheduler,
- gradient calculation,
- training DataLoader shuffling,
- early stopping

must not run in production inference.

Inference must use:

```python
model.eval()
torch.inference_mode()
deterministic preprocessing
```

---

# 70. Model Normalization at Runtime

Backend must use the exact normalization statistics used during training.

For Model 1:

```text
metadata/model1_normalization.json
```

Do not replace with:

```text
ImageNet mean = ...
ImageNet std = ...
```

unless retraining the model accordingly.

For Models 2–4, each model may eventually have its own normalization file if computed separately.

---

# 71. Model Score Handling

Final model outputs are raw logits.

Backend may convert logits to scores using Softmax for display/ranking:

```python
scores = torch.softmax(logits, dim=1)
```

However:

```text
Softmax score != guaranteed calibrated real-world probability
```

Frontend wording should avoid claims such as:

```text
"The model is 99.9% certain biologically"
```

Prefer:

```text
Prediction score: 99.9%
```

with an appropriate disclaimer.

---

# 72. Confidence / Rejection — Future

The original synopsis includes a validation-based uncertainty/rejection mechanism.

Possible future behavior:

- low Model 1 score -> request a better image,
- low Model 2 score -> unsupported/uncertain crop,
- low disease score -> "uncertain — retake image."

Thresholds must be chosen from validation behavior.

Do not hardcode arbitrary thresholds such as 0.50 or 0.90 without validation.

---

# 73. Error Propagation in Hierarchical Models

Hierarchical systems have an important property:

An early error prevents later models from correcting it.

Example:

```text
Actual tomato leaf
Model 1 -> Leaf (correct)
Model 2 -> Potato (wrong)
Model 3 -> Potato Late Blight
```

The final disease output becomes wrong even if Model 4 would have classified it correctly.

Therefore evaluate:

- each model independently,
- full end-to-end pipeline.

Do not derive end-to-end accuracy merely from one model's accuracy.

---

# 74. Backend Logging Recommendations

For development only, log:

- request ID
- preprocessing duration
- Model 1 inference duration
- Model 2 inference duration
- disease model inference duration
- total duration
- predicted class
- scores
- rejection reason

Avoid logging raw uploaded images by default.

Avoid storing private user images unless explicitly needed and disclosed.

---

# 75. Suggested Backend Timing Response — Development Only

Optional debug mode:

```json
{
  "timing_ms": {
    "decode": 8,
    "dsp": 21,
    "model1": 4,
    "model2": 4,
    "disease_model": 5,
    "total": 42
  }
}
```

This should not necessarily be exposed in production UI.

---

# 76. Disease Information Layer

Frontend/backend can later map predicted classes to user-friendly educational information.

Example:

```text
Tomato - Early Blight
```

Could display:

- short explanation,
- common visual symptoms,
- suggested next steps,
- "not a substitute for expert diagnosis."

Disease advice should be sourced carefully and treated as informational.

The ML model itself only classifies; disease management content is a separate product layer.

---

# 77. Image Preview vs DSP Preview

For normal users, show the original uploaded image.

For developer/demo/academic mode, optionally show:

```text
Original
Mask
DSP-processed image
```

This is valuable because DSP is a core academic feature of the project.

Possible frontend toggle:

```text
Show DSP processing
```

This can help demonstrate the project's DSP contribution in presentations.

---

# 78. Academic Demo View

A special page can visually explain:

```text
1. Original RGB image
2. HSV/S-channel representation
3. Otsu mask
4. Morphological cleanup
5. Final masked RGB image
6. Model prediction
7. Future Grad-CAM
```

This is especially useful because the project is for a DSP laboratory, not merely a generic web classifier.

---

# 79. Model 1 Dataset Bias Warning

PlantVillage contains clean leaf images and controlled backgrounds.

Non-leaf examples come from Caltech/Places.

A model could learn dataset-specific differences.

The project already reduces one shortcut through DSP masking and diverse non-leaf sources, but later field testing is still required.

Do not claim universal "leaf detector" robustness based solely on current test performance.

---

# 80. Why PlantVillage Full Color Set Is Used for Model 1

Model 1 should learn generic leaves rather than only potato/tomato leaves.

Therefore using all PlantVillage crop classes for positive Leaf examples improves leaf diversity.

This is intentional.

Models 2–4 later specialize to crop and disease.

---

# 81. Why Non-Leaf Sources Are Combined

Caltech-256 provides:

- objects,
- animals,
- tools,
- household items,
- diverse object-centric images.

Places365 provides:

- indoor scenes,
- man-made scenes,
- environmental backgrounds.

The combination is more diverse than a single small dataset.

---

# 82. Current Edge-Case Policy

The user explicitly decided:

```text
Do not focus on cactus/leaf-like edge cases yet.
```

First objective:

- get the standard pipeline working,
- train/evaluate Model 1,
- then improve failure modes later.

A new chatbot should not derail current work by redesigning Model 1 into multiple edge-case classes unless asked.

---

# 83. Important "Do Not" Decisions

Unless the user changes direction:

- Do not use Jupyter.
- Do not use cloud training.
- Do not use pretrained weights.
- Do not use transfer learning.
- Do not silently replace DSP with raw RGB.
- Do not reintroduce a raw RGB branch unless requested.
- Do not use PlantVillage grayscale/segmented duplicates for Model 1.
- Do not cache DSP images as PNG/WebP for training; benchmark showed it slower.
- Do not remove `cv2.setNumThreads(1)` without rebenchmarking.
- Do not use validation/test augmentation.
- Do not compute normalization from val/test.
- Do not modify original datasets destructively.
- Do not force "Other Plant" into Potato/Tomato.
- Do not assume Step 15 or later training is complete without reports/checkpoints.

---

# 84. Current Step-by-Step Project Status

## Completed

### Step 1
Inspected PlantVillage.

### Step 2
Downloaded Caltech-256.

### Step 3
Audited Caltech categories.

### Step 4
Downloaded/filtered Places365 subset.

### Step 5
Created Model 1 master metadata.

### Step 6
Created leakage-safe 80/10/10 split.

### Step 7
Created deterministic base preprocessing and initial stats.

### Step 8
Implemented DSP pipeline.

### Step 9
Ran large DSP mask audit.

### Step 10
Computed DSP-specific train-only normalization and locked deterministic transform.

### Step 11
Implemented training-only augmentation.

### Step 12
Implemented Dataset/DataLoader and verified batches.

### Step 13
Benchmarked DSP caching and rejected it.

### Step 14
Optimized on-the-fly DataLoader to ~720.8 images/sec.

## Next

### Step 15
Implement and validate ResNet-9 architecture for Model 1.

---

# 85. Next Work After Step 15

Expected next phases:

```text
Step 16
Training engine setup

Step 17
CUDA/AMP/training sanity checks

Step 18
1-epoch benchmark

Step 19
Review loss/accuracy/performance

Step 20
Full Model 1 training

Step 21
Model 1 validation/test evaluation

Step 22
Model 1 inference wrapper

Then:
Model 2 dataset preparation
Model 2 training
Model 3 preparation/training
Model 4 preparation/training

Then:
end-to-end pipeline integration
backend
frontend
explainability
deployment work
```

Step numbering after Step 15 is conceptual and may shift depending on future decisions.

---

# 86. Suggested Training Artifacts

For every model, save:

```text
configuration JSON/YAML
best checkpoint
last checkpoint
training history
validation history
confusion matrix
per-class metrics
training/validation loss graph
training/validation accuracy graph
inference benchmark
```

Possible structure:

```text
results/model1/
results/model2/
results/model3/
results/model4/
```

---

# 87. Suggested Configuration File

Example concept:

```yaml
model_name: model1_resnet9
task: leaf_vs_non_leaf
input_size: 224
num_classes: 2
batch_size: 64
num_workers: 8
prefetch_factor: 4
persistent_workers: true
opencv_threads: 1
random_seed: 42
dsp_enabled: true
```

Training-specific fields should be added after hyperparameters are finalized.

---

# 88. Backend Should Be Checkpoint-Agnostic

Backend config should reference checkpoint locations centrally.

Example:

```python
MODEL1_CHECKPOINT = ...
MODEL2_CHECKPOINT = ...
MODEL3_CHECKPOINT = ...
MODEL4_CHECKPOINT = ...
```

Do not embed absolute development paths throughout backend files.

Use environment variables or application configuration for production.

---

# 89. Suggested Frontend Component Structure

```text
frontend/src/
├── components/
│   ├── UploadDropzone.jsx
│   ├── ImagePreview.jsx
│   ├── AnalyzeButton.jsx
│   ├── LoadingState.jsx
│   ├── PredictionCard.jsx
│   ├── RejectionCard.jsx
│   ├── DiseaseInfoCard.jsx
│   └── DspVisualization.jsx
│
├── pages/
│   ├── Home.jsx
│   ├── About.jsx
│   └── HowItWorks.jsx
│
├── services/
│   └── predictionApi.js
│
├── App.jsx
└── main.jsx
```

---

# 90. Frontend UX States

Frontend state machine:

```text
IDLE
 -> IMAGE_SELECTED
 -> VALIDATING
 -> READY
 -> UPLOADING
 -> ANALYZING
 -> SUCCESS
 -> REJECTED
 -> ERROR
```

Prevent duplicate Analyze requests while one is already running.

---

# 91. Suggested Result Object in Frontend

```javascript
{
  leafDetected: true,
  supported: true,
  crop: "tomato",
  disease: "early_blight",
  displayName: "Tomato - Early Blight",
  scores: {
    leaf: 0.99,
    crop: 0.97,
    disease: 0.94
  }
}
```

Keep backend snake_case vs frontend camelCase conversion consistent.

---

# 92. CORS

During development:

Frontend likely:

```text
http://localhost:5173
```

Backend likely:

```text
http://localhost:8000
```

FastAPI must explicitly allow the frontend origin.

Do not use unrestricted CORS in production without reason.

---

# 93. API Documentation

FastAPI provides:

```text
/docs
```

and:

```text
/redoc
```

Use these during frontend integration.

The backend response schema should remain stable so frontend work can proceed even while models are being retrained.

---

# 94. Mock Backend for Frontend Development

Frontend does not need to wait for all four trained models.

Backend/frontend can initially use mock responses matching final API contracts.

Example mock cases:

- tomato early blight,
- potato healthy,
- non-leaf,
- unsupported crop,
- invalid image.

Later replace mock inference with actual model registry.

This allows parallel development.

---

# 95. Separation Between Product Errors and Prediction Rejections

Do not treat:

```text
non-leaf
unsupported crop
low confidence
```

as server failures.

These are valid prediction outcomes.

Server failures are:

- model missing,
- exception,
- corrupted internal state,
- GPU failure,
- unexpected preprocessing crash.

Frontend should show them differently.

---

# 96. Future Explainability Endpoint

Possible later endpoint:

```text
POST /api/v1/explain
```

or integrate explanation into `/predict`.

Potential outputs:

- Grad-CAM heatmap image,
- overlay image,
- selected model/class.

Only generate explanations when model architecture supports them and output is validated.

---

# 97. Security Considerations

For a public backend:

- enforce max upload size,
- decode images safely,
- do not execute uploaded content,
- randomize temporary filenames,
- remove temporary files,
- rate limit if internet-facing,
- do not trust MIME headers alone,
- avoid arbitrary file paths from requests.

---

# 98. Performance Considerations

Model inference may be fast compared with DSP/image decoding.

Possible backend optimizations later:

- load models once,
- reuse DSP result,
- use `torch.inference_mode()`,
- avoid unnecessary disk writes,
- keep tensors/device transfers controlled,
- optional GPU warm-up,
- batch only if product traffic needs it.

Do not optimize prematurely before measuring.

---

# 99. Current Training Input Size

```text
224 x 224 RGB
```

PlantVillage original:

```text
256 x 256
```

The DSP output is resized/cropped to 224x224.

Any frontend image can be larger; backend handles preprocessing.

Do not force users to upload exactly 224x224.

---

# 100. User-Facing Supported Crop Scope

Current product should explicitly say:

```text
Supported disease analysis:
Potato
Tomato
```

Model 1 may detect leaves from many crops, but disease classification only supports potato/tomato.

This distinction is important.

---

# 101. Current Disease Scope

Supported conditions:

### Potato
- Healthy
- Early Blight
- Late Blight

### Tomato
- Healthy
- Early Blight
- Late Blight

Do not expose PlantVillage's many other tomato diseases as supported unless Model 4 is deliberately expanded later.

---

# 102. Why Model 2 Includes Other Plant

Without an Other Plant class:

```text
Apple leaf
 -> forced Potato/Tomato
 -> false disease result
```

With Other Plant:

```text
Apple leaf
 -> Other Plant
 -> Unsupported
```

This is a major improvement over a closed two-crop classifier.

---

# 103. Original Synopsis Research/Architecture Context

The synopsis retained ResNet-9 because a selected 2023 reference study reported strong results and because it is compact and easy to explain layer by layer.

MobileNetV2 was proposed as the lightweight comparison because depthwise-separable/inverted-residual operations reduce computation and are suitable for edge-oriented deployment.

This remains useful context if the project later reintroduces model comparison.

---

# 104. Planned Explainability Interpretation

Grad-CAM/SHAP should be used to ask:

```text
Is the model attending to leaf tissue / lesions?
or
Is it attending to irrelevant background?
```

It must not be presented as biological proof that highlighted pixels "caused" the disease.

---

# 105. Planned Evaluation Metrics Across Disease Models

For disease models, include:

- accuracy,
- precision,
- recall,
- macro-F1,
- per-class F1,
- confusion matrix,
- latency,
- model size.

Macro-F1 is important because class imbalance exists, especially for potato.

---

# 106. INT8 / Edge Goal

Long-term:

Compare selected model:

```text
FP32
vs
INT8-compatible version
```

Measure:

- accuracy change,
- F1 change,
- file size,
- inference latency.

This is not a current training priority.

---

# 107. Dataset Mutation Policy

Original source datasets must remain untouched.

Do not:

- rename original image files,
- move them destructively,
- resize all originals,
- overwrite them,
- delete excluded files.

Use metadata CSVs for selection and labeling.

This is already the Model 1 approach.

---

# 108. Why Metadata CSVs Are Used

Benefits:

- reproducible datasets,
- no duplicated files,
- easy relabeling,
- easy auditing,
- easy split tracking,
- source provenance,
- prevents accidental destructive reorganization.

Use the same principle for Models 2–4.

---

# 109. Model 2 Metadata Recommendation

Potential file:

```text
metadata/model2_master.csv
```

Columns:

```text
image_path
label
label_id
source
original_class
crop_group
split
```

Do not reuse Model 1 split blindly without deciding whether shared split consistency is desired across hierarchical models.

A deliberate split policy should be documented.

---

# 110. Model 3 / 4 Metadata Recommendation

Potential:

```text
metadata/model3_potato.csv
metadata/model4_tomato.csv
```

Each should preserve:

- image path,
- disease class,
- label id,
- source,
- split.

Use duplicate/leakage checks again.

---

# 111. Potential Full Inference Response

Example:

```json
{
  "success": true,
  "request_id": "uuid",
  "leaf": {
    "label": "leaf",
    "score": 0.995
  },
  "crop": {
    "label": "tomato",
    "score": 0.972
  },
  "disease": {
    "label": "early_blight",
    "display_name": "Early Blight",
    "score": 0.944
  },
  "final": {
    "display_name": "Tomato - Early Blight",
    "supported": true
  },
  "explanation": null
}
```

---

# 112. Naming Convention Recommendation

Use stable machine labels:

```text
non_leaf
leaf

potato
tomato
other

healthy
early_blight
late_blight
```

Use display labels separately:

```text
Non-Leaf
Leaf
Potato
Tomato
Other Plant
Healthy
Early Blight
Late Blight
```

Do not make UI strings the canonical model labels.

---

# 113. Backend Versioning

Use:

```text
/api/v1/
```

This allows future changes without breaking clients.

Potential endpoints:

```text
GET  /api/v1/health
GET  /api/v1/models
POST /api/v1/predict
```

Later:

```text
POST /api/v1/explain
```

---

# 114. Health Endpoint

Example:

```json
{
  "status": "ok",
  "models": {
    "model1": "loaded",
    "model2": "loaded",
    "model3": "loaded",
    "model4": "loaded"
  },
  "device": "cuda"
}
```

Do not expose sensitive local filesystem paths.

---

# 115. Development Milestone Strategy

Recommended milestones:

## Milestone A
Model 1 works independently.

## Milestone B
Model 2 works independently.

## Milestone C
Models 3 and 4 work independently.

## Milestone D
Full local Python inference pipeline.

## Milestone E
FastAPI integration.

## Milestone F
Frontend integration.

## Milestone G
Explainability.

## Milestone H
Field validation / edge optimization.

---

# 116. What a New Chatbot Should Do Next

If receiving this document while the project is still at the current state:

1. Do not redesign the project.
2. Check whether `reports/model1_resnet9_architecture.txt` exists.
3. If Step 15 is incomplete, finish Step 15 only.
4. If Step 15 is complete, inspect its report before defining training hyperparameters.
5. Continue one step at a time.
6. Do not start Models 2–4 until Model 1 has been fully trained and evaluated unless the user explicitly wants parallel work.
7. For frontend/backend work, use the hierarchical API contract described here and mock untrained stages if necessary.

---

# 117. Important Context About Antigravity Workflow

The user is using Antigravity to execute local tasks.

The preferred workflow is:

```text
ChatGPT gives ONE precise prompt
        |
User gives prompt to Antigravity
        |
Antigravity performs one step
        |
User returns report
        |
ChatGPT reviews
        |
Next prompt only
```

The user explicitly does **not** want a huge multi-step Antigravity prompt that runs everything automatically.

Always proceed step by step.

---

# 118. Current Philosophy on Optimization

Measure first.

Examples already followed:

- full PlantVillage used first rather than prematurely sampling,
- DSP mask behavior audited before training,
- cache benchmark run before committing,
- DataLoader worker/thread benchmark run before accepting slow throughput.

Continue this philosophy.

Do not optimize based only on assumptions.

---

# 119. Current Storage Situation

Storage changed during downloads/extractions, but latest report during optimization showed enough free storage for project operation.

A cache was not adopted despite fitting because it was slower.

Before downloading large future datasets or saving huge artifacts, check current free disk space again.

Do not assume older free-space numbers remain accurate.

---

# 120. Final High-Level Summary

The current project is a **DSP-first hierarchical plant disease detection system**.

Core design:

```text
Upload
  |
DSP leaf/foreground isolation
  |
Model 1: Leaf vs Non-Leaf
  |
Model 2: Potato / Tomato / Other
  |
  +-- Potato -> Model 3
  |              Healthy
  |              Early Blight
  |              Late Blight
  |
  +-- Tomato -> Model 4
                 Healthy
                 Early Blight
                 Late Blight
```

Model 1 dataset is fully prepared:

```text
111,087 total images
54,305 Leaf
56,782 Non-Leaf
```

Leakage-safe split:

```text
88,869 train
11,108 val
11,110 test
```

DSP preprocessing is finalized for the first Model 1 experiment.

On-the-fly DSP data throughput was optimized to:

```text
~720.8 images/sec
```

using:

```text
8 workers
persistent workers
prefetch 4
OpenCV 1 thread per worker
```

Model 1 ResNet-9 implementation is the immediate next step.

Frontend/backend can be developed in parallel using mock API responses, but actual production inference must load the exact preprocessing, normalization, label mappings, and checkpoints used during training.

---

# 121. One-Paragraph Context for Another Chatbot

This project is a local PyTorch/CUDA DSP-assisted plant disease detector being built on a Windows Lenovo LOQ with an RTX 5060. It has evolved from the original six-class synopsis into a hierarchical four-model system. Every image first goes through a deterministic OpenCV DSP pipeline using RGB->HSV, S-channel Gaussian blur, Otsu thresholding, 5x5 elliptical morphological opening/closing, connected-components largest-blob filtering, and masking of the original RGB foreground against black, followed by 224x224 resize/crop, train-only augmentation, tensor conversion and DSP-specific train-derived normalization. Model 1 classifies Leaf vs Non-Leaf using 54,305 PlantVillage color images as Leaf and 56,782 filtered Caltech-256 + Places365 images as Non-Leaf, with a SHA-256 leakage-safe 80/10/10 split. The DataLoader is optimized to ~720.8 img/s with 8 workers, persistent workers, prefetch 4 and OpenCV one-thread-per-worker. Model 1 is about to use a manually implemented ResNet-9 trained from random initialization with no transfer learning. Model 2 will classify Potato/Tomato/Other Plant, Model 3 Potato Healthy/Early/Late Blight, and Model 4 Tomato Healthy/Early/Late Blight. Backend should preferably be FastAPI and load models once, reuse deterministic DSP preprocessing, and return rejection states for non-leaf/unsupported crops. Frontend can be React/Vite with drag-drop upload, preview, loading state, prediction/rejection cards, and later DSP/Grad-CAM visualizations. Do not use Jupyter, cloud training, pretrained weights, raw-RGB training, dataset-destructive reorganization, or PNG/WebP DSP caching unless the user changes direction.

---

# 122. Final Handoff Warning

Do not assume every planned component is already implemented.

Confirmed Model 1 data/DSP pipeline work is substantially complete.

At the time of this document:

```text
Model 1 ResNet-9 architecture validation = NEXT STEP / pending report
Model 1 full training = not yet completed
Model 2 training = not started
Model 3 training = not started
Model 4 training = not started
FastAPI backend = not yet confirmed implemented
React frontend = not yet confirmed implemented
Grad-CAM/SHAP = not yet implemented
MobileNetV2 = not yet implemented
INT8 = not yet implemented
field validation = not yet completed
```

Always inspect the actual project files/reports/checkpoints before claiming later stages are complete.
