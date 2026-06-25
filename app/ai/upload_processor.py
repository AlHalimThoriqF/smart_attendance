import time
import cv2
import os
import re
import datetime
import numpy as np
from collections import deque, Counter
from app.database.database import SessionLocal
from app import repositories
from app.ai import recognition

SMOOTH_WINDOW = 5
MIN_SVM_CONF_UPDATE = 0.5
CONFIDENCE_THRESHOLD = 0.6
MAX_TRACK_DISTANCE = 100
MAX_MISS_FRAME = 15       
MAX_VELOCITY = 30
MAX_PREDICT_DISTANCE = 50

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def predict_svm(embedding):
    if recognition.svm_model is None or recognition.label_encoder is None:
        return "Unknown", "Unknown", 0.0, True
        
    X = embedding.reshape(1, -1)
    
    if hasattr(recognition.svm_model, "predict_proba"):
        proba = recognition.svm_model.predict_proba(X)[0]
        prob_sorted = np.sort(proba)[::-1]
        
        max_prob = float(prob_sorted[0])
        second_prob = float(prob_sorted[1]) if len(prob_sorted) > 1 else 0.0
        margin = max_prob - second_prob
        
        pred_encoded = recognition.svm_model.classes_[np.argmax(proba)]
        svm_raw_name = recognition.label_encoder.inverse_transform([pred_encoded])[0]
        svm_conf = max_prob
        
        is_unknown = False
        if margin < 0.05 or svm_conf < CONFIDENCE_THRESHOLD:
            is_unknown = True
            
    else:
        pred_encoded = recognition.svm_model.predict(X)[0]
        svm_raw_name = recognition.label_encoder.inverse_transform([pred_encoded])[0]
        svm_conf = 1.0
        is_unknown = False

    final_name = "Unknown" if is_unknown else svm_raw_name
    return final_name, svm_raw_name, svm_conf, is_unknown

def get_stable_name(track_id, raw_name, svm_conf, tracks):
    history = tracks[track_id]["history"]
    if svm_conf is not None and svm_conf >= MIN_SVM_CONF_UPDATE:
        history.append(raw_name)

    if len(history) == 0:
        return raw_name, 0, 0

    counter = Counter(history)
    stable_name, vote_count = counter.most_common(1)[0]
    return stable_name, vote_count, len(history)

class VideoUploadProcessorManager:
    def __init__(self):
        self.upload_progress = {} # Menyimpan progress proses upload per task_id
        self.last_logged = {} # Menyimpan waktu terakhir log untuk tiap (cctv_id, person_name) agar tidak spam DB
        
    def process_uploaded_video(self, file_path: str, start_time: datetime.datetime, cctv_name: str, cctv_id: int = None, task_id: str = None):
        db = SessionLocal()
            
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 25.0
            
        if task_id:
            self.upload_progress[task_id] = {
                'progress': 0, 
                'status': 'processing', 
                'message': 'Membaca metadata video...',
                'info': {
                    'fps': round(fps, 2),
                    'total_frames': total_frames,
                    'resolution': f"{width}x{height}"
                }
            }
            
        processed_video_filename = None
        processed_video_path = None
        video_writer = None
        
        if task_id:
            videos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "videos")
            os.makedirs(videos_dir, exist_ok=True)
            processed_video_filename = f"{task_id}_processed.mp4"
            processed_video_path = os.path.join(videos_dir, processed_video_filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
            
        print(f"Starting to process uploaded video: {file_path} with start time {start_time}")
        
        # State tracking khusus untuk video ini
        tracks = {}
        next_track_id = 1
        
        def create_new_track(center, bbox):
            nonlocal next_track_id
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
                "actual_center": center,
                # Atribut untuk DB Log
                "best_conf": 0.0,
                "best_frame": None,
                "best_frame_with_boxes": None,
                "best_box": None,
                "timestamp": None
            }
            return track_id

        def assign_track(center, bbox, used_track_ids, current_frame):
            if len(tracks) == 0:
                return create_new_track(center, bbox)

            valid_tracks = {}
            for tid, data in tracks.items():
                if tid in used_track_ids:
                    continue
                if data["last_seen"] == -1 or (current_frame - data["last_seen"]) <= MAX_MISS_FRAME:
                    valid_tracks[tid] = data

            best_track_id = None
            best_dist = 999999
            for track_id, data in valid_tracks.items():
                dist = distance(center, data["center"])
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
                    vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx))
                    vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy))
                    
                    old_vx, old_vy = tracks[best_track_id].get("velocity", (0, 0))
                    tracks[best_track_id]["velocity"] = (0.5 * old_vx + 0.5 * vx, 0.5 * old_vy + 0.5 * vy)

                tracks[best_track_id]["center"] = center
                tracks[best_track_id]["bbox"] = bbox
                tracks[best_track_id]["actual_center"] = center
                return best_track_id

            return create_new_track(center, bbox)

        frame_idx = 0
        try:
            while cap.isOpened():
                if task_id and self.upload_progress.get(task_id, {}).get('status') == 'cancelled':
                    print(f"Upload processing task {task_id} cancelled.")
                    break
                    
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                frame_idx += 1
                
                if task_id and total_frames > 0 and frame_idx % 5 == 0:
                    progress_pct = min(99, int((frame_idx / total_frames) * 100))
                    if task_id in self.upload_progress:
                        self.upload_progress[task_id]['progress'] = progress_pct
                        self.upload_progress[task_id]['message'] = f'Memproses frame {frame_idx}/{total_frames}'
                
                current_time_offset = datetime.timedelta(seconds=frame_idx / fps)
                current_timestamp = start_time + current_time_offset
                
                faces = []
                if recognition.face_analysis:
                    try:
                        faces = recognition.face_analysis.get(frame, max_num=0)
                    except Exception as e:
                        print(f"InsightFace error: {e}")
                
                used_track_ids = []
                for i, face in enumerate(faces, start=1):
                    det_conf = float(face.det_score)
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    bbox = (x1, y1, x2, y2)
                    center = get_center(bbox)

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
                            stable_name, vote_count, total_vote = get_stable_name(track_id, prediksi_asli_svm, svm_conf, tracks)

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

                    if svm_conf is not None:
                        current_max = tracks[track_id].get("max_svm_conf")
                        if current_max is None or svm_conf > current_max:
                            tracks[track_id]["max_svm_conf"] = svm_conf



                    # Simpan snapshot terbaik
                    if svm_conf is not None and svm_conf > tracks[track_id]["best_conf"]:
                        tracks[track_id]["best_conf"] = svm_conf
                        tracks[track_id]["best_frame"] = frame.copy()
                        tracks[track_id]["best_box"] = bbox
                        tracks[track_id]["timestamp"] = current_timestamp

                    label = f"{stable_name}"
                    if svm_conf is not None and stable_name != "Unknown":
                        label += f" ({svm_conf*100:.1f}%)"

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

                # PREDISKI / HOLD UNTUK WAJAH YANG TIDAK TERDETEKSI DI FRAME INI
                for tid, data in tracks.items():
                    if data["last_seen"] >= 0 and data["last_seen"] < frame_idx and (frame_idx - data["last_seen"]) <= MAX_MISS_FRAME:
                        vx, vy = data.get("velocity", (0, 0))
                        
                        actual_cx, actual_cy = data.get("actual_center", data["center"])
                        curr_cx, curr_cy = data["center"]
                        if distance((curr_cx + vx, curr_cy + vy), (actual_cx, actual_cy)) > MAX_PREDICT_DISTANCE:
                            vx, vy = 0, 0  # Stop bergerak jika sudah terlalu jauh
                        
                        x1, y1, x2, y2 = data["bbox"]
                        new_x1, new_y1 = int(x1 + vx), int(y1 + vy)
                        new_x2, new_y2 = int(x2 + vx), int(y2 + vy)
                        
                        new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
                        new_x2, new_y2 = min(width, new_x2), min(height, new_y2)

                        data["bbox"] = (new_x1, new_y1, new_x2, new_y2)
                        data["center"] = get_center(data["bbox"])

                valid_tracks_this_frame = []
                for track_id, data in list(tracks.items()):
                    if data["last_seen"] >= 0:
                        if (frame_idx - data["last_seen"]) <= MAX_MISS_FRAME:
                            valid_tracks_this_frame.append((track_id, data))
                        else:
                            # LOG KE DATABASE KARENA TRACK SUDAH MATI (LEWAT DARI MAX_MISS_FRAME)
                            if data["stable_name"] not in ["Unknown", "embedding_gagal"] and data["best_frame"] is not None:
                                # Buat box di best_frame
                                frame_with_boxes = data["best_frame"].copy()
                                bx1, by1, bx2, by2 = data["best_box"]
                                cv2.rectangle(frame_with_boxes, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                cv2.putText(frame_with_boxes, data["label"], (bx1, max(by1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                                data["best_frame_with_boxes"] = frame_with_boxes
                                
                                self._save_and_log_track(db, cctv_id, cctv_name, data)
                            del tracks[track_id] # Hapus dari memori

                # DRAWING
                frame_draw = frame.copy()
                for track_id, data in valid_tracks_this_frame:
                    x1, y1, x2, y2 = data["bbox"]
                    label = data["label"]
                    stable_name = data["stable_name"]

                    box_color = (0, 255, 0)
                    if stable_name == "Unknown":
                        box_color = (0, 0, 255)

                    cv2.rectangle(frame_draw, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame_draw, label, (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                                
                if video_writer:
                    video_writer.write(frame_draw)
                            
        except Exception as e:
            import traceback
            print(f"CRITICAL ERROR in process_uploaded_video: {e}")
            traceback.print_exc()
        finally:
            # Flush semuan sisa track di buffer yang belum sempat ke log
            for track_id, data in tracks.items():
                if data["stable_name"] not in ["Unknown", "embedding_gagal"] and data["best_frame"] is not None:
                    frame_with_boxes = data["best_frame"].copy()
                    bx1, by1, bx2, by2 = data["best_box"]
                    cv2.rectangle(frame_with_boxes, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(frame_with_boxes, data["label"], (bx1, max(by1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    data["best_frame_with_boxes"] = frame_with_boxes
                    self._save_and_log_track(db, cctv_id, cctv_name, data)
                
            if task_id:
                current_status = self.upload_progress.get(task_id, {}).get('status')
                if current_status != 'cancelled':
                    info_data = self.upload_progress.get(task_id, {}).get('info', {})
                    if processed_video_filename:
                        info_data['processed_video_url'] = f"/videos/{processed_video_filename}"
                        
                    self.upload_progress[task_id] = {
                        'progress': 100,
                        'status': 'completed',
                        'message': 'Pemrosesan selesai!',
                        'info': info_data
                    }
            if video_writer:
                video_writer.release()
            cap.release()
            db.close()
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            print(f"Finished processing uploaded video: {file_path}")

    def _save_and_log_track(self, db, cctv_id, cctv_name, track):
        person_name = track['stable_name']
        frame = track['best_frame']
        frame_with_boxes = track['best_frame_with_boxes']
        confidence = track['best_conf']
        timestamp = track['timestamp']
        x1, y1, x2, y2 = track['best_box']
        
        height, width = frame.shape[:2]
        face_w = x2 - x1
        face_h = y2 - y1
        
        pad_x = int(face_w * 1.5)
        pad_y_top = int(face_h * 0.7)
        pad_y_bottom = int(face_h * 0.7)
        
        start_y = max(0, y1 - pad_y_top)
        end_y = min(height, y2 + pad_y_bottom)
        start_x = max(0, x1 - pad_x)
        end_x = min(width, x2 + pad_x)
        
        snapshot_frame_crop = frame[start_y:end_y, start_x:end_x].copy()
        
        session_uuid = timestamp.strftime("%Y%m%d_%H%M%S")
        folder_name = re.sub(r'[\\/*?:"<>|]', "", person_name).strip()
        base_snapshot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "snapshots")
        user_snapshot_dir = os.path.join(base_snapshot_dir, folder_name)
        os.makedirs(user_snapshot_dir, exist_ok=True)
        
        last_crop_filename = f"upload_{session_uuid}_crop.jpg"
        last_full_filename = f"upload_{session_uuid}_full.jpg"
        
        if snapshot_frame_crop.shape[0] > 0 and snapshot_frame_crop.shape[1] > 0:
            cv2.imwrite(os.path.join(user_snapshot_dir, last_crop_filename), snapshot_frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_path_db = f"{folder_name}/{last_crop_filename}"
        else:
            crop_path_db = None
            
        snapshot_frame_full = frame_with_boxes.copy()
        if width > 1280:
            scale_full = 1280 / width
            snapshot_frame_full = cv2.resize(snapshot_frame_full, (int(width * scale_full), int(height * scale_full)))
        cv2.imwrite(os.path.join(user_snapshot_dir, last_full_filename), snapshot_frame_full, [cv2.IMWRITE_JPEG_QUALITY, 80])
        full_path_db = f"{folder_name}/{last_full_filename}"
        
        # Cegah spam db
        current_seconds = timestamp.timestamp()
        last_time_seen = self.last_logged.get((cctv_id, person_name, "upload"), 0)
        if current_seconds - last_time_seen <= 3:
            return
            
        repositories.logs.create_detection_log(
            db, cctv_id, person_name, confidence, 
            snapshot_path=full_path_db,
            crop_snapshot_path=crop_path_db,
            timestamp_override=timestamp,
            cctv_name=cctv_name
        )
        self.last_logged[(cctv_id, person_name, "upload")] = current_seconds
        print(f"Logged {person_name} from uploaded video at {timestamp} with highest conf {confidence:.2f}")

VideoUploadProcessor = VideoUploadProcessorManager()
