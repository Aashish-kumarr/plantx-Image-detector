"""Real hierarchical PyTorch inference implementation."""
import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image

from app.core.logging import get_logger
from app.ml.interfaces import InferencePipelineInterface, PipelineResult, StageResult

from app.ml.pytorch_models.architectures.model1_resnet9 import ResNet9Model1
from app.ml.pytorch_models.preprocessing.dsp_preprocessing import apply_dsp_preprocessing

logger = get_logger("ml.pytorch_inference")

# Crop class mapping for Model 2
MODEL2_LABELS = {0: "apple", 1: "blueberry", 2: "cherry", 3: "corn", 4: "grape", 
                 5: "orange", 6: "peach", 7: "pepper", 8: "potato", 9: "raspberry", 
                 10: "soybean", 11: "squash", 12: "strawberry", 13: "tomato"}

SINGLE_CONDITION_CROPS = {
    "blueberry": "healthy",
    "orange": "haunglongbing_citrus_greening",
    "raspberry": "healthy",
    "soybean": "healthy",
    "squash": "powdery_mildew"
}

def human_readable(label: str) -> str:
    overrides = {
        "apple_scab": "Apple Scab",
        "cedar_apple_rust": "Cedar Apple Rust",
        "powdery_mildew": "Powdery Mildew",
        "cercospora_leaf_spot_gray_leaf_spot": "Cercospora Leaf Spot / Gray Leaf Spot",
        "common_rust": "Common Rust",
        "northern_leaf_blight": "Northern Leaf Blight",
        "esca_black_measles": "Esca (Black Measles)",
        "leaf_blight_isariopsis_leaf_spot": "Leaf Blight (Isariopsis Leaf Spot)",
        "bacterial_spot": "Bacterial Spot",
        "leaf_scorch": "Leaf Scorch",
        "healthy": "Healthy",
        "early_blight": "Early Blight",
        "late_blight": "Late Blight",
        "leaf_mold": "Leaf Mold",
        "septoria_leaf_spot": "Septoria Leaf Spot",
        "spider_mites": "Spider Mites",
        "target_spot": "Target Spot",
        "tomato_mosaic_virus": "Tomato Mosaic Virus",
        "tomato_yellow_leaf_curl_virus": "Tomato Yellow Leaf Curl Virus",
        "haunglongbing_citrus_greening": "Huanglongbing (Citrus Greening)",
        "huanglongbing_citrus_greening": "Huanglongbing (Citrus Greening)"
    }
    return overrides.get(label, label.replace("_", " ").title())

def safe_load_state_dict(checkpoint):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        return checkpoint["model_state"]
    else:
        return checkpoint

class PyTorchInferencePipeline(InferencePipelineInterface):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorchInferencePipeline initializing on device: {self.device}")
        
        base_dir = os.path.join(os.path.dirname(__file__), "pytorch_models", "weights")
        self.weights_dir = os.path.join(os.path.dirname(__file__), "..", "..", "weights")
        if not os.path.exists(self.weights_dir):
            self.weights_dir = base_dir # fallback if needed
            
        # Model 1 & 2
        self.model1 = ResNet9Model1(3, 2).to(self.device)
        self.model1.load_state_dict(safe_load_state_dict(torch.load(os.path.join(base_dir, "model1_leaf.pth"), map_location=self.device, weights_only=True)), strict=True)
        self.model1.eval()
        self.m1_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.1151, 0.1231, 0.0813], std=[0.2034, 0.2096, 0.1593])
        ])
        
        self.model2 = ResNet9Model1(3, 14).to(self.device)
        self.model2.load_state_dict(safe_load_state_dict(torch.load(os.path.join(base_dir, "model2_crop.pth"), map_location=self.device, weights_only=True)), strict=True)
        self.model2.eval()
        self.m2_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.1257, 0.1572, 0.0856], std=[0.1620, 0.1985, 0.1149])
        ])

        # Disease Model Registry
        self.disease_registry = {
            "potato": {"model_num": 3, "classes": {0: "healthy", 1: "early_blight", 2: "late_blight"}, 
                       "mean": [0.1627, 0.1850, 0.1104], "std": [0.1957, 0.2238, 0.1414], "file": os.path.join(base_dir, "model3_potato.pth")},
            "tomato": {"model_num": 4, "classes": {0: "healthy", 1: "early_blight", 2: "late_blight", 3: "bacterial_spot", 4: "leaf_mold", 5: "septoria_leaf_spot", 6: "spider_mites", 7: "target_spot", 8: "tomato_mosaic_virus", 9: "tomato_yellow_leaf_curl_virus"}, 
                       "mean": [0.1070, 0.1361, 0.0718], "std": [0.1560, 0.1926, 0.1082], "file": os.path.join(base_dir, "model4_tomato.pth")},
            "apple": {"model_num": 5, "classes": {0: "healthy", 1: "apple_scab", 2: "black_rot", 3: "cedar_apple_rust"}, 
                      "mean": [0.07347409, 0.10038915, 0.06589010], "std": [0.13495880, 0.17648672, 0.12671633], "file": os.path.join(self.weights_dir, "model5_apple.pth")},
            "cherry": {"model_num": 6, "classes": {0: "healthy", 1: "powdery_mildew"}, 
                       "mean": [0.14351691, 0.18846236, 0.11188254], "std": [0.18727443, 0.24050464, 0.15725942], "file": os.path.join(self.weights_dir, "model6_cherry.pth")},
            "corn": {"model_num": 7, "classes": {0: "healthy", 1: "cercospora_leaf_spot_gray_leaf_spot", 2: "common_rust", 3: "northern_leaf_blight"}, 
                     "mean": [0.13903937, 0.16719028, 0.10428933], "std": [0.20518272, 0.23928076, 0.16130961], "file": os.path.join(self.weights_dir, "model7_corn.pth")},
            "grape": {"model_num": 8, "classes": {0: "healthy", 1: "black_rot", 2: "esca_black_measles", 3: "leaf_blight_isariopsis_leaf_spot"}, 
                      "mean": [0.13681938, 0.17084016, 0.08285125], "std": [0.18252350, 0.21670182, 0.11859776], "file": os.path.join(self.weights_dir, "model8_grape.pth")},
            "peach": {"model_num": 9, "classes": {0: "healthy", 1: "bacterial_spot"}, 
                      "mean": [0.10522229, 0.10649589, 0.04322645], "std": [0.20703991, 0.19445668, 0.08863733], "file": os.path.join(self.weights_dir, "model9_peach.pth")},
            "pepper": {"model_num": 10, "classes": {0: "healthy", 1: "bacterial_spot"}, 
                       "mean": [0.12141660, 0.16457595, 0.08729180], "std": [0.17571826, 0.22456476, 0.13301843], "file": os.path.join(self.weights_dir, "model10_pepper.pth")},
            "strawberry": {"model_num": 11, "classes": {0: "healthy", 1: "leaf_scorch"}, 
                           "mean": [0.19913457, 0.21676643, 0.11664810], "std": [0.23870961, 0.25625631, 0.15102065], "file": os.path.join(self.weights_dir, "model11_strawberry.pth")}
        }
        
        self.disease_models = {}
        for crop, info in self.disease_registry.items():
            num_classes = len(info["classes"])
            model = ResNet9Model1(3, num_classes).to(self.device)
            state = safe_load_state_dict(torch.load(info["file"], map_location=self.device, weights_only=True))
            model.load_state_dict(state, strict=True)
            model.eval()
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=info["mean"], std=info["std"])
            ])
            self.disease_models[crop] = (model, transform)

    def execute(self, preprocessed_input: bytes, *, scenario: str = "") -> PipelineResult:
        pil_image = Image.open(io.BytesIO(preprocessed_input)).convert("RGB")
        dsp_pil = apply_dsp_preprocessing(pil_image)
        
        with torch.inference_mode():
            # Run Model 1
            t1 = self.m1_transform(dsp_pil).unsqueeze(0).to(self.device)
            prob1 = torch.softmax(self.model1(t1), dim=1)[0]
            m1_conf, m1_idx = torch.max(prob1, dim=0)
            leaf_pred = "leaf" if m1_idx.item() == 1 else "non_leaf"
            
            if leaf_pred == "non_leaf":
                return PipelineResult(
                    status="not_leaf",
                    leaf_check=StageResult(label="non_leaf", score=m1_conf.item()),
                    crop_check=None,
                    prediction=None
                )
                
            # Run Model 2
            t2 = self.m2_transform(dsp_pil).unsqueeze(0).to(self.device)
            prob2 = torch.softmax(self.model2(t2), dim=1)[0]
            m2_conf, m2_idx = torch.max(prob2, dim=0)
            crop_pred = MODEL2_LABELS[m2_idx.item()]
            
            # Single condition crops: return sole PlantVillage condition directly without disease CNN
            if crop_pred in SINGLE_CONDITION_CROPS:
                disease_pred = SINGLE_CONDITION_CROPS[crop_pred]
                status = "healthy" if disease_pred == "healthy" else "disease_detected"
                return PipelineResult(
                    status=status,
                    leaf_check=StageResult(label="leaf", score=m1_conf.item()),
                    crop_check=StageResult(label=crop_pred, score=m2_conf.item()),
                    prediction={
                        "crop": crop_pred,
                        "disease_supported": True,
                        "class_id": disease_pred,
                        "display_name": human_readable(disease_pred),
                        "score": None,
                        "model_version": None,
                        "model_route": {"leaf_model": 1, "crop_model": 2, "disease_model": None},
                        "disease_source": "single_dataset_class"
                    }
                )
                
            # Multi-class crops: run dedicated disease model
            info = self.disease_registry[crop_pred]
            model, transform = self.disease_models[crop_pred]
            
            td = transform(dsp_pil).unsqueeze(0).to(self.device)
            probd = torch.softmax(model(td), dim=1)[0]
            md_conf, md_idx = torch.max(probd, dim=0)
            disease_pred = info["classes"][md_idx.item()]
            
            status = "healthy" if disease_pred == "healthy" else "disease_detected"
            
            return PipelineResult(
                status=status,
                leaf_check=StageResult(label="leaf", score=m1_conf.item()),
                crop_check=StageResult(label=crop_pred, score=m2_conf.item()),
                prediction={
                    "crop": crop_pred,
                    "disease_supported": True,
                    "class_id": disease_pred,
                    "display_name": human_readable(disease_pred),
                    "score": md_conf.item(),
                    "model_version": f"v1.0 (Model {info['model_num']})",
                    "model_route": {"leaf_model": 1, "crop_model": 2, "disease_model": info["model_num"]},
                    "disease_source": "disease_model"
                }
            )

pytorch_inference_pipeline = PyTorchInferencePipeline()
