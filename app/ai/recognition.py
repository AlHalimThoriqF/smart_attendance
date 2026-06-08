import os
import pickle
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from fastapi import HTTPException, status
from app.repositories.lectures import FACES_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "svm_face_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")

svm_model = None
face_analysis = None
label_encoder = None

def initialize_models():
    global svm_model, face_analysis, label_encoder
    
    try:
        face_analysis = FaceAnalysis(
            name='buffalo_sc',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace initialized successfully with execution providers.")
    except Exception as e:
        print(f"Error initializing InsightFace: {e}")
        pass

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                svm_model = pickle.load(f)
            print("Pre-trained SVM model loaded successfully from disk.")
        except Exception as e:
            print(f"Error loading SVM model from {MODEL_PATH}: {e}")
    else:
        print(f"SVM model not found at {MODEL_PATH}. Face recognition will only draw bounding boxes without names.")

    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            with open(LABEL_ENCODER_PATH, "rb") as f:
                label_encoder = pickle.load(f)
            print("Label Encoder loaded successfully from disk.")
        except Exception as e:
            print(f"Error loading Label Encoder from {LABEL_ENCODER_PATH}: {e}")

def predict_frame(img):
    global svm_model, face_analysis, label_encoder
    
    if face_analysis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="InsightFace model is not initialized."
        )

    if img is None:
        return []

    try:
        faces = face_analysis.get(img)
    except Exception as e:
        print(f"InsightFace processing failed: {str(e)}")
        return []

    response_faces = []
    
    if not faces:
        return response_faces

    for face in faces:
        bbox = face.bbox  # np.array([x1, y1, x2, y2])
        box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

        user_id = None
        name = "Unknown"
        confidence = 0.0

        if svm_model is not None and face.embedding is not None:
            embedding = face.embedding.reshape(1, -1)
            try:
                predicted_label = svm_model.predict(embedding)[0]
                
                probabilities = svm_model.predict_proba(embedding)[0]
                class_index = list(svm_model.classes_).index(predicted_label)
                confidence = float(probabilities[class_index])
                
                CONFIDENCE_THRESHOLD = 0.7
                
                if confidence < CONFIDENCE_THRESHOLD:
                    name = "Unknown"
                    user_id = None
                else:
                    # Decode label if label_encoder exists
                    if label_encoder is not None:
                        decoded_label = label_encoder.inverse_transform([predicted_label])[0]

                        name = str(decoded_label)
                        user_id = None
                    else:
                        # Fallback to assuming predicted_label is the user_id integer
                        user_id = int(predicted_label)
                        name = f"User_{user_id}"

            except Exception as e:
                print(f"Classification failure for face embedding: {e}")

        response_faces.append({
            "user_id": user_id,
            "name": name,
            "confidence": confidence,
            "box": box
        })

    return response_faces
