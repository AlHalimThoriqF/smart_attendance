import os
import cv2
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque, Counter
import sys
from insightface.app import FaceAnalysis

# PILIH VIDEO UJI
try:
    VIDEO_PATH = input("Masukkan path/lokasi video (drag & drop file video ke sini, lalu Enter): ").strip(' \'"')
except KeyboardInterrupt:
    print("\nDibatalkan oleh pengguna. Keluar...")
    sys.exit(0)

if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
    print("Video tidak valid atau tidak ditemukan. Keluar...")
    sys.exit(1)

print("Video input:", VIDEO_PATH)

OUTPUT_PATH = "hasil_deteksi_svm3.mp4"
LOG_PATH = "log_deteksi_per_frame3.csv"

# PATH MODEL
SAVE_DIR = "."
SVM_PATH = os.path.join(SAVE_DIR, "svm_face_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")

# LOAD MODEL SVM & LABEL ENCODER
svm_model = joblib.load(SVM_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Model SVM dan LabelEncoder siap.")
print("Kelas:", list(label_encoder.classes_))

# LOAD INSIGHTFACE
MODEL_NAME = "buffalo_sc"

app = FaceAnalysis(
    name=MODEL_NAME,
    providers=["CPUExecutionProvider"]
)
app.prepare(
    ctx_id=0,det_size=(1280, 1280),
)

# PARAMETER

SMOOTH_WINDOW = 5
MIN_SVM_CONF_UPDATE = 0.5
CONFIDENCE_THRESHOLD = 0.6
MAX_TRACK_DISTANCE = 100
MAX_MISS_FRAME = 15       
MAX_VELOCITY = 30
MAX_PREDICT_DISTANCE = 50

tracks = {}
next_track_id = 1

# FUNGSI EMBEDDING & SVM

def predict_svm(embedding):
    X = embedding.reshape(1, -1)
    
    if hasattr(svm_model, "predict_proba"):
        proba = svm_model.predict_proba(X)[0]
        prob_sorted = np.sort(proba)[::-1]
        
        max_prob = float(prob_sorted[0])
        second_prob = float(prob_sorted[1]) if len(prob_sorted) > 1 else 0.0
        margin = max_prob - second_prob
        
        pred_encoded = svm_model.classes_[np.argmax(proba)]
        svm_raw_name = label_encoder.inverse_transform([pred_encoded])[0]
        svm_conf = max_prob
        
        is_unknown = False
        if margin < 0.05 or svm_conf < CONFIDENCE_THRESHOLD:
            is_unknown = True
            
    else:
        pred_encoded = svm_model.predict(X)[0]
        svm_raw_name = label_encoder.inverse_transform([pred_encoded])[0]
        svm_conf = 1.0
        is_unknown = False

    final_name = "Unknown" if is_unknown else svm_raw_name

    return final_name, svm_raw_name, svm_conf, is_unknown

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def create_new_track(center, bbox):
    global next_track_id
    track_id = next_track_id
    next_track_id += 1

    tracks[track_id] = {
        "center": center,
        "bbox": bbox,
        "history": deque(maxlen=SMOOTH_WINDOW),
        "last_seen": -1,
        "label": "",
        "raw_name": "",
        "stable_name": "",
        "locked_name": None,
        "svm_conf": None,
        "max_svm_conf": None,
        "det_conf": None,
        "vote": 0,
        "total_vote": 0,
        "velocity": (0, 0),
        "actual_center": center
    }
    return track_id

def assign_track(center, bbox, used_track_ids, current_frame):
    if len(tracks) == 0:
        return create_new_track(center, bbox)

    # Hanya cek ID yang masih "hidup" atau baru saja hilang sejenak
    valid_tracks = {}
    for tid, data in tracks.items():
        if tid in used_track_ids:
            continue
        if data["last_seen"] == -1 or (current_frame - data["last_seen"]) <= MAX_MISS_FRAME:
            valid_tracks[tid] = data

    best_track_id = None

    # Tracking Fisik berdasarkan jarak terdekat
    best_dist = 999999
    for track_id, data in valid_tracks.items():
        dist = distance(center, data["center"])

        # Jika jaraknya dalam batas maksimal yang diizinkan
        if dist <= MAX_TRACK_DISTANCE:
            if dist < best_dist:
                best_dist = dist
                best_track_id = track_id

    if best_track_id is not None:
        old_cx, old_cy = tracks[best_track_id].get("actual_center", tracks[best_track_id]["center"])
        dt = current_frame - tracks[best_track_id]["last_seen"]
        
        if dt > 0:
            vx = (center[0] - old_cx) / dt
            vy = (center[1] - old_cy) / dt
            
            # Batasi kecepatan maksimal per frame
            vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx))
            vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy))
            
            old_vx, old_vy = tracks[best_track_id].get("velocity", (0, 0))
            tracks[best_track_id]["velocity"] = (0.5 * old_vx + 0.5 * vx, 0.5 * old_vy + 0.5 * vy)

        tracks[best_track_id]["center"] = center
        tracks[best_track_id]["bbox"] = bbox
        tracks[best_track_id]["actual_center"] = center
        return best_track_id

    # Jika tidak ada yang cocok, berarti ini objek baru
    return create_new_track(center, bbox)

def get_stable_name(track_id, raw_name, svm_conf):
    history = tracks[track_id]["history"]
    if svm_conf is not None and svm_conf >= MIN_SVM_CONF_UPDATE:
        history.append(raw_name)

    if len(history) == 0:
        return raw_name, 0, 0

    counter = Counter(history)
    stable_name, vote_count = counter.most_common(1)[0]
    return stable_name, vote_count, len(history)


# =========================
# BUKA VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Video tidak bisa dibuka.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("FPS:", fps)
print("Resolusi:", width, "x", height)
print("Total frame:", total_frames)

# =========================
# PROSES VIDEO
# =========================
frame_idx = 0
log_rows = []

pbar = tqdm(total=total_frames, desc="Memproses video")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    faces = app.get(frame, max_num=0)
    used_track_ids = []

    # --- PROSES DETEKSI & TRACKING ---
    for i, face in enumerate(faces, start=1):
        det_conf = float(face.det_score)
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        bbox = (x1, y1, x2, y2)
        center = get_center(bbox)

        # Assign track ID secara murni spasial
        track_id = assign_track(center, bbox, used_track_ids, frame_idx)
        used_track_ids.append(track_id)

        emb = face.normed_embedding
        if emb is not None:
            emb = emb.flatten().astype(np.float32)

        raw_name = "embedding_gagal"
        stable_name = "embedding_gagal"
        prediksi_asli_svm = None
        is_unknown = False
        svm_conf = None
        vote_count = tracks[track_id]["vote"]
        total_vote = tracks[track_id]["total_vote"]

        if emb is not None:
            raw_name, prediksi_asli_svm, svm_conf, is_unknown = predict_svm(emb)

            if tracks[track_id]["locked_name"] is not None:
                stable_name = tracks[track_id]["locked_name"]
                raw_name = stable_name
            else:
                stable_name, vote_count, total_vote = get_stable_name(track_id, prediksi_asli_svm, svm_conf)

                if is_unknown:
                    stable_name = "Unknown"
                    raw_name = "Unknown"

            if stable_name not in ["Unknown", "embedding_gagal"]:
                if tracks[track_id]["locked_name"] is None:
                    if svm_conf is not None and svm_conf >= CONFIDENCE_THRESHOLD:
                        tracks[track_id]["locked_name"] = raw_name
                        stable_name = raw_name
                    elif vote_count >= 5:
                        tracks[track_id]["locked_name"] = stable_name

        # Update nilai tertinggi svm_conf (tetap disimpan untuk keperluan log, jika ada)
        if svm_conf is not None:
            current_max = tracks[track_id].get("max_svm_conf")
            if current_max is None or svm_conf > current_max:
                tracks[track_id]["max_svm_conf"] = svm_conf

        label = f"{stable_name}"

        # Update memori track
        tracks[track_id]["center"] = center
        tracks[track_id]["bbox"] = bbox
        tracks[track_id]["last_seen"] = frame_idx
        tracks[track_id]["label"] = label
        tracks[track_id]["raw_name"] = raw_name
        tracks[track_id]["stable_name"] = stable_name
        tracks[track_id]["svm_conf"] = svm_conf
        tracks[track_id]["det_conf"] = det_conf
        tracks[track_id]["vote"] = vote_count
        tracks[track_id]["total_vote"] = total_vote
        tracks[track_id]["prediksi_asli_svm"] = prediksi_asli_svm

    # PREDISKI / HOLD UNTUK WAJAH YANG TIDAK TERDETEKSI DI FRAME INI
    for tid, data in tracks.items():
        if data["last_seen"] >= 0 and data["last_seen"] < frame_idx and (frame_idx - data["last_seen"]) <= MAX_MISS_FRAME:
            vx, vy = data.get("velocity", (0, 0))
            
            # Batasi jarak tebakan agar tidak terlempar terlalu jauh dari posisi asli terakhir
            actual_cx, actual_cy = data.get("actual_center", data["center"])
            curr_cx, curr_cy = data["center"]
            if distance((curr_cx + vx, curr_cy + vy), (actual_cx, actual_cy)) > MAX_PREDICT_DISTANCE:
                vx, vy = 0, 0  # Stop bergerak jika sudah terlalu jauh
            
            x1, y1, x2, y2 = data["bbox"]
            
            # Apply velocity
            new_x1, new_y1 = int(x1 + vx), int(y1 + vy)
            new_x2, new_y2 = int(x2 + vx), int(y2 + vy)
            
            # Keep inside frame bounds
            new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
            new_x2, new_y2 = min(width, new_x2), min(height, new_y2)

            data["bbox"] = (new_x1, new_y1, new_x2, new_y2)
            data["center"] = get_center(data["bbox"])

    valid_tracks_this_frame = []
    for track_id, data in tracks.items():
        if data["last_seen"] >= 0 and (frame_idx - data["last_seen"]) <= MAX_MISS_FRAME:
            valid_tracks_this_frame.append((track_id, data))

    jumlah_deteksi_stabil = len(valid_tracks_this_frame)
    
    # Info frame
    info_frame = f"Frame: {frame_idx} | Detected: {jumlah_deteksi_stabil}"
    cv2.putText(frame, info_frame, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if jumlah_deteksi_stabil == 0:
        log_rows.append({
            "frame": frame_idx, "face_ke": 0, "track_id": None, "jumlah_deteksi_frame": 0,
            "nama_prediksi_raw": None, "nama_prediksi_stabil": None, "prediksi_asli_svm": None, "is_unknown": None,
            "confidence_deteksi": None, "confidence_svm": None, "vote": None, "total_vote": None, "status_bbox": "no_detection",
            "x1": None, "y1": None, "x2": None, "y2": None
        })
    else:
        for i, (track_id, data) in enumerate(valid_tracks_this_frame, start=1):
            miss = frame_idx - data["last_seen"]
            status_bbox = "detected" if miss == 0 else "hold"
            
            x1, y1, x2, y2 = data["bbox"]
            label = data["label"]
            stable_name = data["stable_name"]

            log_rows.append({
                "frame": frame_idx, "face_ke": i, "track_id": track_id, "jumlah_deteksi_frame": jumlah_deteksi_stabil,
                "nama_prediksi_raw": data["raw_name"], "nama_prediksi_stabil": stable_name, 
                "prediksi_asli_svm": data.get("prediksi_asli_svm", None), "is_unknown": stable_name == "Unknown",
                "confidence_deteksi": data["det_conf"], "confidence_svm": data["svm_conf"], 
                "vote": data["vote"], "total_vote": data["total_vote"], "status_bbox": status_bbox,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

            # WARNA DEFAULT ADALAH HIJAU
            box_color = (0, 255, 0)
            if stable_name == "Unknown":
                box_color = (0, 0, 255) # BGR: Merah

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    out.write(frame)
    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()
print("Video selesai:", OUTPUT_PATH)

# SIMPAN LOG CSV
df_log = pd.DataFrame(log_rows)
df_log.to_csv(LOG_PATH, index=False)

print("Log deteksi disimpan:", LOG_PATH)
print("Total baris CSV:", len(df_log))
print(df_log.head(20))
