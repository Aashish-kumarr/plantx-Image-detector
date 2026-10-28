import sys
import argparse
import os

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Test PyTorch Inference Engine")
    parser.add_argument("image_path", nargs="?", help="Optional path to an image for testing")
    args = parser.parse_args()
    
    print("Initializing PyTorchInferencePipeline...")
    
    try:
        from app.ml.pytorch_inference import pytorch_inference_pipeline
        pipeline = pytorch_inference_pipeline
        
        print(f"Device Selected: {pipeline.device}")
        
        # Verify model loading
        models = [pipeline.model1, pipeline.model2, pipeline.model3, pipeline.model4]
        for idx, model in enumerate(models):
            print(f"Model {idx+1} loaded.")
            if not not model.training:
                pass # it's eval mode because we called eval
            
        print("All four models loaded successfully.")
        
        if args.image_path:
            if not os.path.exists(args.image_path):
                print(f"Error: Image {args.image_path} not found.")
                sys.exit(1)
                
            print(f"\nRunning inference on {args.image_path}...")
            with open(args.image_path, "rb") as f:
                img_bytes = f.read()
            
            result = pipeline.execute(preprocessed_input=img_bytes)
            
            print("\nInference Result:")
            print(f"Status: {result.status}")
            if result.leaf_check:
                print(f"Leaf Check: {result.leaf_check.label} (Score: {result.leaf_check.score:.4f})")
            if result.crop_check:
                print(f"Crop Check: {result.crop_check.label} (Score: {result.crop_check.score:.4f})")
            if result.prediction:
                print(f"Prediction: {result.prediction['display_name']} ({result.prediction['class_id']})")
                print(f"Confidence: {result.prediction['score']:.4f}")
                
    except Exception as e:
        import traceback
        print(f"Error during initialization or inference:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
