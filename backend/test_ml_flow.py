import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app.main import app

def test_inference(image_path):
    print(f"Testing real inference with {image_path}")
    
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    
    import mongomock
    from unittest.mock import patch
    with patch("app.db.client.MongoClient", mongomock.MongoClient), TestClient(app) as client:
    
        # Fake user id due to auth, let's bypass auth if possible or create a mock.
        # We can just override the dependency.
        from app.core.auth import get_current_user
        from app.core.auth import AuthUser
        
        def override_get_current_user():
            return AuthUser(uid="test_user_123", email="test@test.com")
            
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        # 1. Upload image
        print("Uploading image...")
        response = client.post(
            "/api/v1/images",
            files={"image": ("test_leaf.jpg", file_bytes, "image/jpeg")}
        )
        if response.status_code != 201:
            print(f"Failed to upload: {response.text}")
            return
            
        data = response.json()
        image_id = data["image_id"]
        print(f"Image uploaded successfully. ID: {image_id}")
        
        # 2. Run analysis
        print("Running analysis...")
        response = client.post(
            "/api/v1/analyses",
            json={"image_id": image_id}
        )
        
        if not response.is_success:
            print(f"Analysis failed: {response.text}")
            return
            
        result = response.json()
        print("Analysis Result:")
        import json
        print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    test_inference(args.image_path)
