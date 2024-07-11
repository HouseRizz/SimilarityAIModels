from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')

# Function to extract features using a pre-trained ResNet model
def extract_features(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(weights=True)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.squeeze().numpy()

def extract_features_from_catalog_images(catalog_dir):
    catalog_features = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    for category in os.listdir(catalog_dir):
        category_path = os.path.join(catalog_dir, category)
        if os.path.isdir(category_path):  # Ensure it's a directory
            for img_path in os.listdir(category_path):
                full_img_path = os.path.join(category_path, img_path)
                if os.path.isfile(full_img_path) and os.path.splitext(full_img_path)[1].lower() in image_extensions:  # Ensure it's an image file
                    features = extract_features(full_img_path)
                    catalog_features[f"{category}/{img_path}"] = features

    print("Features extracted for all catalog images.")
    return catalog_features

def find_similar_items(features, catalog_features, top_k=5):
    similarities = {}
    for img_path, catalog_features in catalog_features.items():
        similarity = cosine_similarity([features], [catalog_features])[0][0]
        similarities[img_path] = similarity

    # Sort the images by similarity score in descending order and get the top_k items
    similar_items = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return similar_items

# Preload catalog features
catalog_dir = '/app/images'
catalog_features = extract_features_from_catalog_images(catalog_dir)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Similarity API"}

@app.post("/detect_and_find_similar")
async def detect_and_find_similar(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    os.makedirs('uploads', exist_ok=True)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Perform object detection and crop objects
    results = yolo_model(source=file_location)
    cropped_images_dir = 'cropped_images'
    os.makedirs(cropped_images_dir, exist_ok=True)
    feature_dict = {}

    for i, result in enumerate(results):
        img = result.orig_img
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(cropped_images_dir, f"cropped_{i}_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)
            features = extract_features(cropped_img_path)
            feature_dict[f"cropped_{i}_{j}.jpg"] = features

    # Generate recommendations
    recommendations = {}
    for img_name, features in feature_dict.items():
        similar_items = find_similar_items(features, catalog_features)
        # Convert numpy array to list before storing in recommendations
        recommendations[img_name] = [(item[0], float(item[1])) for item in similar_items]

    # Return JSONResponse with custom JSON encoder
    return JSONResponse(content=recommendations, indent=2, default=json_default)

# Custom JSON Encoder function to handle numpy arrays
def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
