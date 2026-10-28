# PlantDx: DSP-Assisted Hierarchical Plant Disease Detection System

A production-grade, full-stack AI platform for crop leaf disease diagnostics. PlantDx uses a cascading hierarchy of 11 lightweight ResNet-9 convolutional neural networks combined with deterministic Digital Signal Processing (DSP) foreground extraction.

---

## 🌟 Highlights & Architecture

The system operates across three tiers:
- **Frontend**: Next.js 16 (React 19) + Tailwind CSS responsive web application
- **Backend**: FastAPI (Python 3.11) + MongoDB async REST service
- **Machine Learning**: 11 ResNet-9 models trained from scratch with deterministic DSP preprocessing

### Hierarchical Decision Pipeline

```text
Input Leaf Image
     │
     ▼
[Product Quality Gate] (Resolution, Blur, Exposure, Contrast) ──► Failed? Return 'quality_failed'
     │ Passed
     ▼
[Deterministic DSP Preprocessing] (RGB → HSV → S-channel Otsu Mask → Aspect-preserving 224x224 crop)
     │
     ▼
[Model 1: Leaf vs. Non-Leaf] ──► non_leaf? Stop early (status = 'not_leaf')
     │ leaf
     ▼
[Model 2: 14-Crop Classifier]
     │
     ├─── Supported Multi-Class Disease Crops:
     │      ├─ Apple       ──► Model 5  (healthy, apple_scab, black_rot, cedar_apple_rust)
     │      ├─ Cherry      ──► Model 6  (healthy, powdery_mildew)
     │      ├─ Corn        ──► Model 7  (healthy, cercospora_leaf_spot, common_rust, northern_leaf_blight)
     │      ├─ Grape       ──► Model 8  (healthy, black_rot, esca_black_measles, leaf_blight)
     │      ├─ Peach       ──► Model 9  (healthy, bacterial_spot)
     │      ├─ Pepper      ──► Model 10 (healthy, bacterial_spot)
     │      ├─ Potato      ──► Model 3  (healthy, early_blight, late_blight)
     │      ├─ Strawberry  ──► Model 11 (healthy, leaf_scorch)
     │      └─ Tomato      ──► Model 4  (10 disease/healthy classes)
     │
     └─── Static Single-Condition Crops (Resolved directly, no disease CNN needed):
            ├─ Blueberry   ──► Healthy
            ├─ Orange      ──► Huanglongbing (Citrus Greening)
            ├─ Raspberry   ──► Healthy
            ├─ Soybean     ──► Healthy
            └─ Squash      ──► Powdery Mildew
```

---

## 📁 Repository Structure

```
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/routes/       # Health, Images, Quality, Analyses REST routes
│   │   ├── core/             # Settings, auth, errors, logging
│   │   ├── db/               # MongoDB client & indexing
│   │   ├── ml/               # Real PyTorch inference pipeline & registries
│   │   │   └── pytorch_models/  # ResNet-9 architectures & DSP transforms
│   │   ├── repositories/     # Data persistence
│   │   ├── schemas/          # Pydantic v2 schemas
│   │   ├── services/         # Image quality & analysis orchestration
│   │   └── storage/          # Local disk image store
│   ├── weights/              # Deployment models manifest & metadata
│   └── pyproject.toml        # Backend dependencies
│
├── frontend/                 # Next.js Web Application
│   ├── app/                  # App Router pages (/app/analyze, /app/history, /app/results/[id])
│   ├── components/           # UI components (analysis, results, dashboard, layout)
│   ├── lib/                  # API client & auth provider
│   └── package.json          # Frontend dependencies
│
├── ml/                       # Machine Learning Training & Evaluation
│   ├── src/                  # PyTorch Dataset loaders, architectures, trainers
│   ├── metadata/             # Split configurations, normalization stats & mappings
│   ├── reports/              # Pretraining, refinement, evaluation & E2E audit reports
│   └── *.py                  # Standalone training, refinement & evaluation scripts
│
├── docs/                     # Architecture & API specifications
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ & npm
- Python 3.11+
- MongoDB running locally on port 27017

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Backend Swagger documentation will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The application will be accessible at: [http://localhost:3000](http://localhost:3000)

---

## 🔬 Model Manifest & Verification

Model metrics, class mappings, and SHA-256 integrity checksums are recorded in [`backend/weights/models_manifest.json`](backend/weights/models_manifest.json). Checkpoint binaries (`*.pth`) can be placed into `backend/weights/` and `backend/app/ml/pytorch_models/weights/`.

---

## 📄 License
MIT License.
