
import os
import pickle
import warnings
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from fastapi import HTTPException, status

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
FACES_DIR = os.path.join(BASE_DIR, "storage", "faces")

MODEL_PATH = os.path.join(ROOT_DIR, "svm_face_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")
CENTERS_PATH = os.path.join(ROOT_DIR, "class_centers.pkl")

svm_model = None
face_analysis = None
label_encoder = None
class_centers = None

def initialize_models():
    # Menginisialisasi model InsightFace, SVM, Label Encoder, dan Centroid ke dalam memori.
    global svm_model, face_analysis, label_encoder, class_centers
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            face_analysis = FaceAnalysis(
                name='buffalo_sc',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            face_analysis.prepare(ctx_id=0, det_size=(1280, 1280))
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

        if os.path.exists(CENTERS_PATH):
            try:
                with open(CENTERS_PATH, "rb") as f:
                    class_centers = pickle.load(f)
                print("Class Centers loaded successfully from disk.")
            except Exception as e:
                print(f"Error loading Class Centers from {CENTERS_PATH}: {e}")

def predict_frame(img):
    # Memproses frame gambar untuk mendeteksi wajah dan memprediksi identitasnya.
    global svm_model, face_analysis, label_encoder, class_centers
    
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
        
        # TRIK PENYELAMAT: Paksa koordinat tetap berada di dalam area frame video
        frame_h, frame_w = img.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame_w, int(bbox[2]))
        y2 = min(frame_h, int(bbox[3]))
        
        box = [x1, y1, x2, y2]

        user_id = None
        name = "Unknown"
        confidence = 0.0

        if svm_model is not None and face.embedding is not None:
            raw_emb = face.embedding.flatten()
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                raw_emb = raw_emb / norm
            embedding = raw_emb.reshape(1, -1)
            
            try:
                probabilities = svm_model.predict_proba(embedding)[0]
                prob_sorted = np.sort(probabilities)[::-1]
                
                max_prob = float(prob_sorted[0])
                second_prob = float(prob_sorted[1]) if len(prob_sorted) > 1 else 0.0
                margin = max_prob - second_prob
                
                CONFIDENCE_THRESHOLD = 0.6
                
                # Jalankan "Satpam Margin" murni untuk filter per orang
                if margin < 0.05 or max_prob < CONFIDENCE_THRESHOLD:
                    name = "Unknown"
                    user_id = None
                    confidence = 0.0 if margin < 0.05 else max_prob
                else:
                    confidence = max_prob
                    # Ambil prediksi nama dari SVM
                    class_index = np.argmax(probabilities)
                    predicted_label = svm_model.classes_[class_index]
                    
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
