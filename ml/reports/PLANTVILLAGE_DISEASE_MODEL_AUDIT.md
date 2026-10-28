# PlantVillage Disease Model Audit

## 1. Dataset Location
`C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\plantvillage dataset\color`

## 2. Dataset Total
Total Images: 54305
Total Class Folders: 38

## 3. Complete Class-Folder Table
| Folder | Crop | Condition | Image Count |
| --- | --- | --- | --- |
| Apple___Apple_scab | Apple | Apple_scab | 630 |
| Apple___Black_rot | Apple | Black_rot | 621 |
| Apple___Cedar_apple_rust | Apple | Cedar_apple_rust | 275 |
| Apple___healthy | Apple | healthy | 1645 |
| Blueberry___healthy | Blueberry | healthy | 1502 |
| Cherry_(including_sour)___Powdery_mildew | Cherry | Powdery_mildew | 1052 |
| Cherry_(including_sour)___healthy | Cherry | healthy | 854 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn | Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Corn_(maize)___Common_rust_ | Corn | Common_rust_ | 1192 |
| Corn_(maize)___Northern_Leaf_Blight | Corn | Northern_Leaf_Blight | 985 |
| Corn_(maize)___healthy | Corn | healthy | 1162 |
| Grape___Black_rot | Grape | Black_rot | 1180 |
| Grape___Esca_(Black_Measles) | Grape | Esca_(Black_Measles) | 1383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape | Leaf_blight_(Isariopsis_Leaf_Spot) | 1076 |
| Grape___healthy | Grape | healthy | 423 |
| Orange___Haunglongbing_(Citrus_greening) | Orange | Haunglongbing_(Citrus_greening) | 5507 |
| Peach___Bacterial_spot | Peach | Bacterial_spot | 2297 |
| Peach___healthy | Peach | healthy | 360 |
| Pepper,_bell___Bacterial_spot | Pepper | Bacterial_spot | 997 |
| Pepper,_bell___healthy | Pepper | healthy | 1478 |
| Potato___Early_blight | Potato | Early_blight | 1000 |
| Potato___Late_blight | Potato | Late_blight | 1000 |
| Potato___healthy | Potato | healthy | 152 |
| Raspberry___healthy | Raspberry | healthy | 371 |
| Soybean___healthy | Soybean | healthy | 5090 |
| Squash___Powdery_mildew | Squash | Powdery_mildew | 1835 |
| Strawberry___Leaf_scorch | Strawberry | Leaf_scorch | 1109 |
| Strawberry___healthy | Strawberry | healthy | 456 |
| Tomato___Bacterial_spot | Tomato | Bacterial_spot | 2127 |
| Tomato___Early_blight | Tomato | Early_blight | 1000 |
| Tomato___Late_blight | Tomato | Late_blight | 1909 |
| Tomato___Leaf_Mold | Tomato | Leaf_Mold | 952 |
| Tomato___Septoria_leaf_spot | Tomato | Septoria_leaf_spot | 1771 |
| Tomato___Spider_mites Two-spotted_spider_mite | Tomato | Spider_mites Two-spotted_spider_mite | 1676 |
| Tomato___Target_Spot | Tomato | Target_Spot | 1404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato | Tomato_Yellow_Leaf_Curl_Virus | 5357 |
| Tomato___Tomato_mosaic_virus | Tomato | Tomato_mosaic_virus | 373 |
| Tomato___healthy | Tomato | healthy | 1591 |

## 4. & 5. Crop-wise Grouping & Image Counts
### APPLE
- Total Images: 3171
- Number of Conditions: 4
  - Apple_scab: 630
  - Black_rot: 621
  - Cedar_apple_rust: 275
  - healthy: 1645

### BLUEBERRY
- Total Images: 1502
- Number of Conditions: 1
  - healthy: 1502

### CHERRY
- Total Images: 1906
- Number of Conditions: 2
  - Powdery_mildew: 1052
  - healthy: 854

### CORN
- Total Images: 3852
- Number of Conditions: 4
  - Cercospora_leaf_spot Gray_leaf_spot: 513
  - Common_rust_: 1192
  - Northern_Leaf_Blight: 985
  - healthy: 1162

### GRAPE
- Total Images: 4062
- Number of Conditions: 4
  - Black_rot: 1180
  - Esca_(Black_Measles): 1383
  - Leaf_blight_(Isariopsis_Leaf_Spot): 1076
  - healthy: 423

### ORANGE
- Total Images: 5507
- Number of Conditions: 1
  - Haunglongbing_(Citrus_greening): 5507

### PEACH
- Total Images: 2657
- Number of Conditions: 2
  - Bacterial_spot: 2297
  - healthy: 360

### PEPPER
- Total Images: 2475
- Number of Conditions: 2
  - Bacterial_spot: 997
  - healthy: 1478

### POTATO
- Total Images: 2152
- Number of Conditions: 3
  - Early_blight: 1000
  - Late_blight: 1000
  - healthy: 152

### RASPBERRY
- Total Images: 371
- Number of Conditions: 1
  - healthy: 371

### SOYBEAN
- Total Images: 5090
- Number of Conditions: 1
  - healthy: 5090

### SQUASH
- Total Images: 1835
- Number of Conditions: 1
  - Powdery_mildew: 1835

### STRAWBERRY
- Total Images: 1565
- Number of Conditions: 2
  - Leaf_scorch: 1109
  - healthy: 456

### TOMATO
- Total Images: 18160
- Number of Conditions: 10
  - Bacterial_spot: 2127
  - Early_blight: 1000
  - Late_blight: 1909
  - Leaf_Mold: 952
  - Septoria_leaf_spot: 1771
  - Spider_mites Two-spotted_spider_mite: 1676
  - Target_Spot: 1404
  - Tomato_Yellow_Leaf_Curl_Virus: 5357
  - Tomato_mosaic_virus: 373
  - healthy: 1591

## 6. & 7. Data Integrity & Duplicates
- Unreadable/corrupted images: 0
- Zero-byte images: 0
- Unsupported formats: 0
- Exact duplicates (extra copies): 21
- Cross-class duplicates (same image in multiple classes): 0

## 8. Existing Model-2 Dataset Consistency Check
The existing `metadata/model2_14class_split.csv` file references images located in the exact same `dataset/plantvillage dataset/color` directory structure. Path prefix matches confirm the consistency of this dataset source.

## 9. Crops Suitable for Disease Classification
- Apple
- Cherry
- Corn
- Grape
- Peach
- Pepper
- Potato
- Strawberry
- Tomato

## 10. Crops Currently Unsuitable
- Blueberry (NOT SUITABLE FOR A STANDALONE DISEASE CLASSIFIER WITH CURRENT DATASET)
- Orange (NOT SUITABLE FOR A STANDALONE DISEASE CLASSIFIER WITH CURRENT DATASET)
- Raspberry (NOT SUITABLE FOR A STANDALONE DISEASE CLASSIFIER WITH CURRENT DATASET)
- Soybean (NOT SUITABLE FOR A STANDALONE DISEASE CLASSIFIER WITH CURRENT DATASET)
- Squash (NOT SUITABLE FOR A STANDALONE DISEASE CLASSIFIER WITH CURRENT DATASET)

## 11. & 12. Proposed Models 5–11 & Exact Class Mappings
### Model 5: APPLE
- Crop: Apple
- Classes: 4
- Total Images: 3171
  - Apple___Apple_scab: 630 images
  - Apple___Black_rot: 621 images
  - Apple___Cedar_apple_rust: 275 images
  - Apple___healthy: 1645 images

### Model 6: CHERRY
- Crop: Cherry
- Classes: 2
- Total Images: 1906
  - Cherry_(including_sour)___Powdery_mildew: 1052 images
  - Cherry_(including_sour)___healthy: 854 images

### Model 7: CORN
- Crop: Corn
- Classes: 4
- Total Images: 3852
  - Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot: 513 images
  - Corn_(maize)___Common_rust_: 1192 images
  - Corn_(maize)___Northern_Leaf_Blight: 985 images
  - Corn_(maize)___healthy: 1162 images

### Model 8: GRAPE
- Crop: Grape
- Classes: 4
- Total Images: 4062
  - Grape___Black_rot: 1180 images
  - Grape___Esca_(Black_Measles): 1383 images
  - Grape___Leaf_blight_(Isariopsis_Leaf_Spot): 1076 images
  - Grape___healthy: 423 images

### Model 9: PEACH
- Crop: Peach
- Classes: 2
- Total Images: 2657
  - Peach___Bacterial_spot: 2297 images
  - Peach___healthy: 360 images

### Model 10: PEPPER
- Crop: Pepper
- Classes: 2
- Total Images: 2475
  - Pepper,_bell___Bacterial_spot: 997 images
  - Pepper,_bell___healthy: 1478 images

### Model 11: STRAWBERRY
- Crop: Strawberry
- Classes: 2
- Total Images: 1565
  - Strawberry___Leaf_scorch: 1109 images
  - Strawberry___healthy: 456 images

## 13. Risks Discovered
- **Data Duplication**: Exact duplicates within the same class artificially inflate dataset size and may cause train/test leakage if split randomly without deduplication.
