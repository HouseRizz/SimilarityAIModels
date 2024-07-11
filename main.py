
import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# Function to mount Google Drive



# Function to check YOLO setup
def check_yolo_setup():
    from IPython.display import display, Image as IPImage
    import ultralytics
    ultralytics.checks()


# Function to initialize YOLO model
def initialize_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)


# Function to extract features using a pre-trained ResNet model
def extract_features(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(pretrained=True)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.squeeze().numpy()


# Function to perform object detection and save cropped images
def detect_and_crop_objects(model, image_url, cropped_images_dir='cropped_images', conf=0.8):
    results = model(source=image_url, conf=conf)
    os.makedirs(cropped_images_dir, exist_ok=True)

    for i, result in enumerate(results):
        img = result.orig_img
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(cropped_images_dir, f"cropped_{i}_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)
            print(f"Cropped image saved to {cropped_img_path}")
    return results


# Function to extract features from cropped images
def extract_features_from_cropped_images(cropped_images_dir='cropped_images'):
    feature_dict = {}
    for img_path in os.listdir(cropped_images_dir):
        full_img_path = os.path.join(cropped_images_dir, img_path)
        if os.path.isfile(full_img_path):  # Ensure it's a file
            features = extract_features(full_img_path)
            feature_dict[img_path] = features
    print("Features extracted for all cropped images.")
    return feature_dict


# Function to extract features from catalog images
def extract_features_from_catalog_images(catalog_dir):
    catalog_features = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    for category in os.listdir(catalog_dir):
        category_path = os.path.join(catalog_dir, category)
        if os.path.isdir(category_path):  # Ensure it's a directory
            for img_path in os.listdir(category_path):
                full_img_path = os.path.join(category_path, img_path)
                if os.path.isfile(full_img_path) and os.path.splitext(full_img_path)[
                    1].lower() in image_extensions:  # Ensure it's an image file
                    features = extract_features(full_img_path)
                    catalog_features[f"{category}/{img_path}"] = features
    print("Features extracted for all catalog images.")
    return catalog_features


# Function to find similar items
def find_similar_items(features, catalog_features, top_k=5):
    similarities = []
    for img_path, catalog_feature in catalog_features.items():
        similarity = cosine_similarity(features.reshape(1, -1), catalog_feature.reshape(1, -1))[0][0]
        similarities.append((img_path, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# Function to generate recommendations
def generate_recommendations(feature_dict, catalog_features):
    recommendations = {}
    for img_name, features in feature_dict.items():
        similar_items = find_similar_items(features, catalog_features)
        recommendations[img_name] = similar_items
    print("Recommendations generated.")
    return recommendations


# Function to display recommendations
def display_recommendations(recommendations, cropped_images_dir, catalog_dir):
    for img_name, similar_items in recommendations.items():
        cropped_img_path = os.path.join(cropped_images_dir, img_name)
        cropped_img = Image.open(cropped_img_path)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, len(similar_items) + 1, 1)
        plt.imshow(cropped_img)
        plt.title("Detected Object")
        plt.axis('off')

        for i, (item_img, similarity) in enumerate(similar_items, start=2):
            item_img_path = os.path.join(catalog_dir, item_img)
            item_img = Image.open(item_img_path)
            plt.subplot(1, len(similar_items) + 1, i)
            plt.imshow(item_img)
            plt.title(f"Similar Item {i - 1}\nSimilarity: {similarity:.2f}")
            plt.axis('off')

        plt.show()


# Now you can call and test each function separately

# Example usage:

check_yolo_setup()
yolo_model = initialize_yolo()

# Perform object detection and crop objects
results = detect_and_crop_objects(yolo_model, "https://i.pinimg.com/564x/63/06/6b/63066b20ad95c537334e9e4885bc07f2.jpg")

# Extract features from cropped images
feature_dict = extract_features_from_cropped_images()
print(f"Extracted features from cropped images: {len(feature_dict)}")

# Extract features from catalog images
catalog_features = extract_features_from_catalog_images('/Users/manya./PycharmProjects/Model/BaseSimilarityModel/images')

print(f"Extracted features from catalog images: {len(catalog_features)}")

# Generate recommendations
recommendations = generate_recommendations(feature_dict, catalog_features)
print(f"Generated recommendations for {len(recommendations)} detected objects")

# Display recommendations
display_recommendations(recommendations, 'cropped_images', '/Users/manya./PycharmProjects/Model/BaseSimilarityModel/images')

