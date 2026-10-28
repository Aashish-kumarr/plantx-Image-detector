# Models 5-11 Backend Integration Report

## 1. Copied Champion Sources & Destination
- **Model 5 (Apple)**
  - Source: `checkpoints/model5_apple_final_refine/best_final_refined_model.pth`
  - Destination: `backend/weights/model5_apple.pth`
  - SHA-256 Match: `True`
- **Model 6 (Cherry)**
  - Source: `checkpoints/model6_cherry/best_model.pth`
  - Destination: `backend/weights/model6_cherry.pth`
  - SHA-256 Match: `True`
- **Model 7 (Corn)**
  - Source: `checkpoints/model7_corn_refine/best_refined_model.pth`
  - Destination: `backend/weights/model7_corn.pth`
  - SHA-256 Match: `True`
- **Model 8 (Grape)**
  - Source: `checkpoints/model8_grape/best_model.pth`
  - Destination: `backend/weights/model8_grape.pth`
  - SHA-256 Match: `True`
- **Model 9 (Peach)**
  - Source: `checkpoints/model9_peach/best_model.pth`
  - Destination: `backend/weights/model9_peach.pth`
  - SHA-256 Match: `True`
- **Model 10 (Pepper)**
  - Source: `checkpoints/model10_pepper/best_model.pth`
  - Destination: `backend/weights/model10_pepper.pth`
  - SHA-256 Match: `True`
- **Model 11 (Strawberry)**
  - Source: `checkpoints/model11_strawberry/best_model.pth`
  - Destination: `backend/weights/model11_strawberry.pth`
  - SHA-256 Match: `True`

## 2. Checkpoint State Keys Discovered
During loading, it was verified that checkpoints generally use `model_state_dict` or `model_state`. A generic, strict state dict extractor `safe_load_state_dict` was implemented in the backend to ensure safe extraction across different training engines used in the project without silent partial loading.

## 3. Disease Model Registry & Routing Table
A generic registry `disease_registry` was implemented in `PyTorchInferencePipeline`. It caches all loaded model instances in memory during `__init__` rather than loading per-request.

**Routing logic:**
- **Leaf check**: Model 1 predicts `leaf` or `non_leaf`
- **Crop check**: Model 2 predicts the crop (1 of 14 classes)
- **Supported Disease Routing:**
  - `apple` → Model 5
  - `cherry` → Model 6
  - `corn` → Model 7
  - `grape` → Model 8
  - `peach` → Model 9
  - `pepper` → Model 10
  - `potato` → Model 3
  - `strawberry` → Model 11
  - `tomato` → Model 4

**Unsupported Crop Behavior:**
- `blueberry`, `orange`, `raspberry`, `soybean`, `squash`
- Output explicitly marks `disease_supported=False`
- Human readable message: "Disease classification is not currently supported for this crop."
- `disease_model` in the route is returned as `None`

## 4. API Response Schema Changes
The API contract and MongoDB persistence schema were extended seamlessly to support the new inference fields by extending the `Prediction` schema in `app/schemas/analysis.py`.
- Added `disease_supported: bool`
- Added `model_route: ModelRoute` (contains `leaf_model`, `crop_model`, `disease_model`)
- Made `class_id`, `display_name`, `score` optional/nullable for the unsupported crop edge case.
Existing analysis document structure is maintained. Legacy records are inherently compatible as missing fields simply deserialize into `None`.

## 5. Startup Load Results & Output Dimensions
- Model 1: Loaded successfully (2 classes)
- Model 2: Loaded successfully (14 classes)
- Model 3 (Potato): Loaded successfully (3 classes)
- Model 4 (Tomato): Loaded successfully (10 classes)
- Model 5 (Apple): Loaded successfully (4 classes)
- Model 6 (Cherry): Loaded successfully (2 classes)
- Model 7 (Corn): Loaded successfully (4 classes)
- Model 8 (Grape): Loaded successfully (4 classes)
- Model 9 (Peach): Loaded successfully (2 classes)
- Model 10 (Pepper): Loaded successfully (2 classes)
- Model 11 (Strawberry): Loaded successfully (2 classes)

All disease models correctly instantiated `ResNet9Model1(3, num_classes)`.

## 6. Smoke Test Results
A robust routing smoke test `backend_smoke_test.py` was executed directly against the pipeline.
- All non-leaf and supported crop routing branches worked.
- Models gracefully applied custom DSP preprocessing and normalization.
- All 5 unsupported crops returned `disease_supported=False`.
- JSON serialization verification explicitly confirmed the new extended `AnalysisResponse` parses the response successfully.

## 7. No Test Sets Consumed
**CONFIRMATION:** None of the test set images from Models 5-11 were loaded or processed during this backend integration. The smoke test relied entirely on structurally correct in-memory image tensors (e.g. `Image.new()`), guaranteeing the locked status of test metrics.
