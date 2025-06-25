import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
from utils import load_facenet_model, get_embedding

# Path to FaceNet model
MODEL_PATH = "facenet_model/20180402-114759.pb"

# Load FaceNet model and face detector
graph = load_facenet_model(MODEL_PATH)
detector = MTCNN()

# Dataset path
DATASET_DIR = "dataset"

# Lists to hold embeddings and labels
embeddings = []
labels = []

# Loop through each person folder
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        results = detector.detect_faces(image)
        if results:
            x, y, w, h = results[0]['box']
            face = image[y:y+h, x:x+w]
            emb = get_embedding(face, graph)
            embeddings.append(emb)
            labels.append(person_name)

# Convert to arrays
X = np.array(embeddings)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split train/test (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train classifier (MLP = Multi-layer perceptron)
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Save trained model and label encoder
os.makedirs("trained_model", exist_ok=True)
dump(clf, "trained_model/classifier.joblib")
dump(encoder, "trained_model/label_encoder.joblib")

print("✅ Model and label encoder saved!")

