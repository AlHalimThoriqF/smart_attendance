import os
import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque, Counter
import sys

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# =========================
# PILIH VIDEO UJI
# =========================
try:
    VIDEO_PATH = input("Masukkan path/lokasi video (drag & drop file video ke sini, lalu Enter): ").strip(' \'"')
except KeyboardInterrupt:
    print("\nDibatalkan oleh pengguna. Keluar...")
    sys.exit(0)

if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
    print("Video tidak valid atau tidak ditemukan. Keluar...")
    sys.exit(1)

print("Video input:", VIDEO_PATH)

OUTPUT_PATH = "hasil_deteksi_svm2.mp4"
LOG_PATH = "log_deteksi_per_frame3.csv"

# =========================
# PATH MODEL
# =========================
# Pastikan model berada di direktori yang sama, atau sesuaikan SAVE_DIR
SAVE_DIR = "."
SVM_PATH = os.path.join(SAVE_DIR, "svm_face_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")
CENTERS_PATH = os.path.join(SAVE_DIR, "galeri_embeding.pkl")

# =========================
# LOAD MODEL SVM & LABEL ENCODER
# =========================
if "svm_model" not in globals():
    svm_model = joblib.load(SVM_PATH)

if "label_encoder" not in globals():
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

class_centers = None
if os.path.exists(CENTERS_PATH):
    import pickle
    with open(CENTERS_PATH, "rb") as f:
        class_centers = pickle.load(f)

print("Model SVM dan LabelEncoder siap.")
print("Kelas:", list(label_encoder.classes_))

# =========================
# LOAD INSIGHTFACE
# =========================
MODEL_NAME = "buffalo_s"

if "app" in globals():
    del app

app = FaceAnalysis(
    name=MODEL_NAME,
    providers=["CPUExecutionProvider"] # Removed CUDAExecutionProvider to prevent warning
)

app.prepare(
    ctx_id=0,
    det_size=(1280, 1280)
)

recognition_model = app.models["recognition"]
print("InsightFace siap:", MODEL_NAME)


# =========================
# PARAMETER
# =========================
SMOOTH_WINDOW = 5
MIN_SVM_CONF_UPDATE = 0.25
UNKNOWN_THRESHOLD = 0.4
LOCK_THRESHOLD = 0.6
MAX_TRACK_DISTANCE = 200
MAX_MISS_FRAME = 25        
MIN_FACE_SIMILARITY = 0.35 
REID_SIMILARITY_THRESHOLD = 0.4
COSINE_THRESHOLD = 0.25 # Ambang batas cosine similarity rendah

tracks = {}
next_track_id = 1

# =========================
# FUNGSI EMBEDDING & SVM
# =========================
def cosine_similarity(emb1, emb2):
    # Fungsi baru untuk mengecek kemiripan wajah
    if emb1 is None or emb2 is None:
        return 0.0
    
    # Jika emb2 berisi kumpulan banyak embedding (misal dimensi (N, 512))
    # Ambil nilai rata-ratanya (centroid) terlebih dahulu
    emb2 = np.array(emb2)
    if len(emb2.shape) > 1:
        emb2 = np.mean(emb2, axis=0)
        
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def get_embedding_from_face(frame, face):
    aligned = norm_crop(
        frame,
        landmark=face.kps,
        image_size=112
    )
    emb = recognition_model.get_feat(aligned)
    emb = emb.flatten().astype(np.float32)

    norm = np.linalg.norm(emb)
    if norm == 0:
        return None

    emb = emb / norm
    return emb

def predict_svm(embedding):
    X = embedding.reshape(1, -1)
    pred_encoded = svm_model.predict(X)[0]
    svm_raw_name = label_encoder.inverse_transform([pred_encoded])[0]

    svm_conf = None
    if hasattr(svm_model, "predict_proba"):
        proba = svm_model.predict_proba(X)[0]
        svm_conf = float(np.max(proba))

    is_unknown = False
    final_name = svm_raw_name

    if svm_conf is not None and svm_conf < UNKNOWN_THRESHOLD:
        is_unknown = True

    if is_unknown:
        final_name = "Unknown"

    return final_name, svm_raw_name, None, svm_conf, None, is_unknown

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def create_new_track(center, bbox, emb=None):
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
        "last_emb": emb,
        "svm_conf": None,
        "cos_sim": None,
        "det_conf": None,
        "vote": 0,
        "total_vote": 0
    }
    return track_id

def assign_track(center, bbox, emb, used_track_ids, current_frame):
    if len(tracks) == 0:
        return create_new_track(center, bbox, emb)

    # Hanya cek ID yang masih "hidup" atau baru saja hilang sejenak
    valid_tracks = {}
    for tid, data in tracks.items():
        if tid in used_track_ids:
            continue
        if data["last_seen"] == -1 or (current_frame - data["last_seen"]) <= MAX_MISS_FRAME:
            valid_tracks[tid] = data

    best_track_id = None

    # TAHAP 1: Tracking Fisik & Jarak (Prioritas Utama)
    best_dist = 999999
    for track_id, data in valid_tracks.items():
        dist = distance(center, data["center"])
        sim = cosine_similarity(emb, data["last_emb"])

        # Syarat 1: Jaraknya masih masuk akal (<= 200 pixel)
        if dist <= MAX_TRACK_DISTANCE:
            # Syarat 2: Sangat dekat (< 80px) ATAU wajahnya minimal mirip
            if dist < 80 or emb is None or data["last_emb"] is None or sim >= MIN_FACE_SIMILARITY:
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id

    if best_track_id is not None:
        tracks[best_track_id]["center"] = center
        tracks[best_track_id]["bbox"] = bbox
        if emb is not None:
            tracks[best_track_id]["last_emb"] = emb
        return best_track_id

    # TAHAP 2: Re-Identification (Mencegah ID copot saat orang bergerak kilat)
    best_sim = -1
    for track_id, data in valid_tracks.items():
        sim = cosine_similarity(emb, data["last_emb"])
        if sim >= REID_SIMILARITY_THRESHOLD and sim > best_sim:
            best_sim = sim
            best_track_id = track_id

    if best_track_id is not None:
        tracks[best_track_id]["center"] = center
        tracks[best_track_id]["bbox"] = bbox
        if emb is not None:
            tracks[best_track_id]["last_emb"] = emb
        return best_track_id

    # TAHAP 3: Jika semua gagal, berarti ini orang baru
    return create_new_track(center, bbox, emb)

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
    raise Exception("Video tidak bisa dibuka.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

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
    if not ret:
        break

    faces = app.get(frame, max_num=0)
    jumlah_deteksi = len(faces)
    used_track_ids = []

    # Info frame
    info_frame = f"Frame: {frame_idx} | Detected: {jumlah_deteksi}"
    cv2.putText(frame, info_frame, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if jumlah_deteksi == 0:
        log_rows.append({
            "frame": frame_idx, "face_ke": 0, "track_id": None, "jumlah_deteksi_frame": 0,
            "nama_prediksi_raw": None, "nama_prediksi_stabil": None, "prediksi_asli_svm": None, "prediksi_asli_cosine": None, "is_unknown": None,
            "confidence_deteksi": None, "confidence_svm": None, "cosine_sim": None, "vote": None, "total_vote": None, "status_bbox": "no_detection",
            "x1": None, "y1": None, "x2": None, "y2": None
        })

    # --- TAHAP 1: KUMPULKAN WAJAH (PELACAKAN MURNI FISIK) ---
    frame_faces_data = []
    for i, face in enumerate(faces, start=1):
        det_conf = float(face.det_score)
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        bbox = (x1, y1, x2, y2)
        center = get_center(bbox)

        emb = get_embedding_from_face(frame, face)

        # Assign track ID di sini, tanpa menggunakan raw_name atau svm_conf!
        track_id = assign_track(center, bbox, emb, used_track_ids, frame_idx)
        used_track_ids.append(track_id)

        frame_faces_data.append({
            "face_ke": i,
            "det_conf": det_conf,
            "bbox": bbox,
            "center": center,
            "emb": emb,
            "track_id": track_id
        })

# --- TAHAP 2: URUTKAN BERDASARKAN ID & TEBAK NAMA (SVM) ---
    frame_faces_data.sort(key=lambda x: x["track_id"])

    claimed_names_this_frame = set()

    for data in frame_faces_data:
        i = data["face_ke"]
        det_conf = data["det_conf"]
        bbox = data["bbox"]
        x1, y1, x2, y2 = bbox
        center = data["center"]
        emb = data["emb"]
        track_id = data["track_id"]

        raw_name = "embedding_gagal"
        stable_name = "embedding_gagal"
        prediksi_asli_svm = None
        prediksi_asli_cosine = None
        is_unknown = False
        svm_conf = None
        cos_sim_val = None
        vote_count = 0
        total_vote = 0

        if emb is not None:
            # Baru tebak SVM di sini
            raw_name, prediksi_asli_svm, prediksi_asli_cosine, svm_conf, cos_sim_val, is_unknown = predict_svm(emb)

            if tracks[track_id]["locked_name"] is not None:
                stable_name = tracks[track_id]["locked_name"]
                raw_name = stable_name
                vote_count = tracks[track_id]["vote"]
                total_vote = tracks[track_id]["total_vote"]
            else:
                # Belum terkunci, jalankan sistem voting seperti biasa
                # Menggunakan prediksi_asli_svm agar voting mencatat tebakan asli, bukan "Unknown"
                stable_name, vote_count, total_vote = get_stable_name(track_id, prediksi_asli_svm, svm_conf)

                # Prioritas 1: Unknown Filter
                if is_unknown:
                    stable_name = "Unknown"
                    raw_name = "Unknown"

            # --- FITUR BARU: MENCEGAH NAMA GANDA DI SATU FRAME ---
            if stable_name not in ["Unknown", "embedding_gagal"]:
                if stable_name in claimed_names_this_frame:
                    # Jika nama sudah diklaim oleh ID yang lebih kecil, paksa ID ini menjadi Unknown
                    stable_name = "Unknown"
                    raw_name = "Unknown"
                else:
                    # Tandai nama ini sudah terpakai di frame ini
                    claimed_names_this_frame.add(stable_name)

                    # --- MODIFIKASI: KUNCI (LOCK) JIKA CONFIDENCE >= 0.6 ATAU VOTE >= 10 ---
                    if tracks[track_id]["locked_name"] is None:
                        # Syarat 1: Confidence (Probabilitas) SVM >= 0.6
                        if svm_conf is not None and svm_conf >= 0.6:
                            tracks[track_id]["locked_name"] = raw_name
                            stable_name = raw_name
                        
                        # Syarat 2: Jika vote count dari nama stabil mencapai 10
                        elif vote_count >= 5:
                            tracks[track_id]["locked_name"] = stable_name

        # FORMAT LABEL DENGAN ID#
        lock_status = " [L]" if tracks[track_id]["locked_name"] else ""
        if svm_conf is not None:
            label = f"{stable_name}{lock_status} | SVM:{svm_conf:.2f}"
            if cos_sim_val is not None:
                label += f" | Cos:{cos_sim_val:.2f}"
        else:
            label = f"{stable_name}"

        # Update memori track
        tracks[track_id]["center"] = center
        tracks[track_id]["bbox"] = bbox
        tracks[track_id]["last_seen"] = frame_idx
        tracks[track_id]["label"] = label
        tracks[track_id]["raw_name"] = raw_name
        tracks[track_id]["stable_name"] = stable_name
        tracks[track_id]["svm_conf"] = svm_conf
        tracks[track_id]["cos_sim"] = cos_sim_val
        tracks[track_id]["det_conf"] = det_conf
        tracks[track_id]["vote"] = vote_count
        tracks[track_id]["total_vote"] = total_vote

        log_rows.append({
            "frame": frame_idx, "face_ke": i, "track_id": track_id, "jumlah_deteksi_frame": jumlah_deteksi,
            "nama_prediksi_raw": raw_name, "nama_prediksi_stabil": stable_name, "prediksi_asli_svm": prediksi_asli_svm, "prediksi_asli_cosine": prediksi_asli_cosine, "is_unknown": is_unknown,
            "confidence_deteksi": det_conf, "confidence_svm": svm_conf, "cosine_sim": cos_sim_val, "vote": vote_count, "total_vote": total_vote, "status_bbox": "detected",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    # =========================
    # GAMBAR BBOX & LOGIC WARNA
    # =========================
    for track_id, data in tracks.items():
        if data["last_seen"] < 0:
            continue

        miss = frame_idx - data["last_seen"]
        if miss > MAX_MISS_FRAME:
            continue

        x1, y1, x2, y2 = data["bbox"]
        label = data["label"]
        stable_name = data["stable_name"]

        # WARNA DEFAULT ADALAH HIJAU
        box_color = (0, 255, 0)

        # JIKA UNKNOWN = MERAH
        if stable_name == "Unknown":
            box_color = (0, 0, 255) # BGR: Merah

        if miss == 0:
            status_text = ""
        else:
            status_text = " | hold"
            # Note: hold tetap menggunakan box_color yang aktif (Hijau/Merah)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label + status_text, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    out.write(frame)
    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()
print("Video sementara selesai:", OUTPUT_PATH)

# =========================
# SIMPAN LOG CSV
# =========================
df_log = pd.DataFrame(log_rows)
df_log.to_csv(LOG_PATH, index=False)

print("Log deteksi disimpan:", LOG_PATH)
print("Total baris CSV:", len(df_log))
print(df_log.head(20))
