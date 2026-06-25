import threading
import time
import cv2
import base64
import uuid
import os
import re
import datetime
from app.database.database import SessionLocal
from app import repositories
from app.ai.recognition import predict_frame


class CameraStreamReader:
    def __init__(self, rtsp_url):
        # Inisialisasi koneksi ke stream url dan mulai thread pembacaan frame.
        self.kamera = cv2.VideoCapture(rtsp_url)
        self.ret = False
        self.frame = None
        self.running = True
        self.is_video_file = isinstance(rtsp_url, str) and not rtsp_url.startswith(('rtsp://', 'http://', 'https://')) and not str(rtsp_url).isdigit()
        
        if self.kamera.isOpened():
            self.ret, self.frame = self.kamera.read()
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

    def update(self):
        # Terus membaca frame dari stream kamera selama thread berjalan
        while self.running:
            if self.kamera.isOpened():
                ret, frame = self.kamera.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
                    
                    # Jika menggunakan rekaman, beri jeda agar video diputar pada 25 FPS (1/25 = 0.04 detik)
                    if self.is_video_file:
                        time.sleep(0.04)
                else:
                    # Jika video file habis, putar kembali dari awal (Looping)
                    if self.is_video_file:
                        self.kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.04)
                    else:
                        time.sleep(1) # Tunggu kamera yang mungkin terputus
            else:
                time.sleep(1)

    def read(self):
        # Mengembalikan status pembacaan dan frame terbaru
        return self.ret, self.frame
        
    def release(self):
        # Menghentikan thread pembacaan dan melepaskan koneksi kamera.
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.kamera.release()

class BackgroundMonitorManager:
    def __init__(self):
        # Inisialisasi penyimpanan thread, event stop, dan frame terbaru untuk tiap CCTV.
        self.active_threads = {}
        self.stop_events = {}
        self.latest_frames = {} 
        self.last_logged = {}
        self.last_snapshot = {} 
        self.session_uuids = {} 
        self.last_update_time = time.time()

    def get_latest_frame(self, cctv_id):
        # Mengambil frame terbaru dari CCTV tertentu yang sudah di-encode base64.
        return self.latest_frames.get(cctv_id)

    def start_camera(self, cctv_id, rtsp_url):
        # Memulai proses monitoring background untuk satu kamera pada thread terpisah.
        if cctv_id in self.active_threads:
            return # Already running
            
        stop_event = threading.Event()
        self.stop_events[cctv_id] = stop_event
        
        thread = threading.Thread(
            target=self._camera_worker,
            args=(cctv_id, rtsp_url, stop_event),
            daemon=True
        )
        self.active_threads[cctv_id] = thread
        thread.start()
        print(f"Background monitoring started for CCTV {cctv_id}")

    def stop_camera(self, cctv_id):
        # Menghentikan proses monitoring background untuk kamera tertentu.
        if cctv_id in self.stop_events:
            self.stop_events[cctv_id].set()
            if cctv_id in self.active_threads:
                self.active_threads[cctv_id].join(timeout=2.0)
                del self.active_threads[cctv_id]
            del self.stop_events[cctv_id]
            print(f"Background monitoring stopped for CCTV {cctv_id}")

    def start_all(self):
        # Memulai monitoring background untuk semua kamera yang berstatus aktif di config.
        from app.config.cctv_config import get_all_cctvs
        cameras = get_all_cctvs()
        for camera in cameras:
            if camera['status']:
                rtsp = camera['rtsp_url']
                if str(rtsp).isdigit():
                    rtsp = int(rtsp)
                self.start_camera(camera['id'], rtsp)

    def _camera_worker(self, cctv_id, rtsp_url, stop_event):
        # Worker utama untuk membaca stream, melakukan deteksi wajah, tracking, dan logging.
        db = SessionLocal()
        stream_obj = CameraStreamReader(rtsp_url)
        
        last_ai_time = 0
        cached_valid_detections = []
        cached_faces_detected = []
        cached_clean_frame = None
        
        fps_start_time = time.time()
        fps_frames_count = 0
        current_fps = 0.0
        
        try:
            while not stop_event.is_set():
                ret, frame = stream_obj.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_copy = frame.copy()
                current_time = time.time()
                
                # Add timestamp watermark
                now = datetime.datetime.now()
                days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
                months = ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November", "Desember"]
                day_name = days[now.weekday()]
                month_name = months[now.month - 1]
                timestamp_str = f"{day_name}, {now.day:02d} {month_name} {now.year} - {now.strftime('%H:%M:%S')} WIB"
                
                text_x = 15
                text_y = 30
                cv2.putText(frame_copy, timestamp_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame_copy, timestamp_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Calculate FPS
                fps_frames_count += 1
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_frames_count / (current_time - fps_start_time)
                    fps_start_time = current_time
                    fps_frames_count = 0
                
                # Display FPS on top right
                fps_str = f"FPS: {current_fps:.1f}"
                frame_height, frame_width = frame_copy.shape[:2]
                text_size = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                fps_x = frame_width - text_size[0] - 15
                cv2.putText(frame_copy, fps_str, (fps_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame_copy, fps_str, (fps_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # AI Inference (Throttled to 2 FPS)
                if current_time - last_ai_time >= 0.5:
                    try:
                        cached_faces_detected = predict_frame(frame_copy)
                        cached_clean_frame = frame.copy()
                    except Exception as e:
                        print(f"AI prediction error on camera {cctv_id}: {e}")
                    
                    last_ai_time = current_time
                    valid_detections = []
                    
                    for face in cached_faces_detected:
                        if not face.get("box"):
                            continue
                        name_label = face.get("name", "Unknown")
                        confidence = face.get("confidence", 0.0)
                        
                        if name_label.lower() != "unknown":
                            valid_detections.append({
                                "confidence": confidence,
                                "box": face.get("box"),
                                "name": name_label
                            })
                    cached_valid_detections = valid_detections

                # Realtime display using cached detections
                for face in cached_faces_detected:
                    if not face.get("box"):
                        continue
                    x1, y1, x2, y2 = face.get("box")
                    
                    name_label = face.get("name", "Unknown")
                    for vd in cached_valid_detections:
                        if vd["box"] == face.get("box"):
                            name_label = vd["name"]
                            break
                            
                    confidence = face.get("confidence", 0.0)
                    color = (0, 0, 255) if name_label.lower() == "unknown" else (0, 255, 0)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    
                    if name_label.lower() == "unknown" or name_label == "UNKNOWN":
                        label = "Unknown"
                    else:
                        # Clean name: strip titles after comma and shorten to max 2 words
                        clean_name = name_label.split(',')[0].strip()
                        words = clean_name.split()
                        if len(words) > 2:
                            clean_name = f"{words[0]} {words[1]}"
                        
                        # Abbreviate second word if still too long
                        if len(clean_name) > 15 and len(words) >= 2:
                            clean_name = f"{words[0]} {words[1][0]}."
                            
                        label = f"{clean_name} ({confidence*100:.1f}%)" if confidence > 0 else clean_name
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.55
                    thickness = 1
                    (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw black outline text for readability without solid background
                    cv2.putText(frame_copy, label, (x1 + 5, y1 - 6), font, font_scale, (0, 0, 0), thickness + 2)
                    # Draw white text
                    cv2.putText(frame_copy, label, (x1 + 5, y1 - 6), font, font_scale, (255, 255, 255), thickness)

                # Database Logging
                needs_logging = []
                for det in cached_valid_detections:
                    person_name = det["name"]
                    last_time = self.last_logged.get((cctv_id, person_name), 0)
                    if current_time - last_time > 3:
                        needs_logging.append(det)
                        
                if needs_logging:
                    height, width = frame_copy.shape[:2]
                    base_snapshot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "snapshots")
                    
                    for det in needs_logging:
                        person_name = det["name"]
                        confidence = det["confidence"]
                        
                        # Zoom to face with padding
                        x1, y1, x2, y2 = det["box"]
                        face_w = x2 - x1
                        face_h = y2 - y1
                        
                        # Tighter padding for a clean, professional profile picture look
                        pad_x = int(face_w * 1.5)        # Kanan-kiri ditambah 30% dari lebar wajah
                        pad_y_top = int(face_h * 0.7)    # Atas ditambah 50% untuk area rambut
                        pad_y_bottom = int(face_h * 0.7) # Bawah ditambah 30% untuk leher/bahu
                        
                        start_y = max(0, y1 - pad_y_top)
                        end_y = min(height, y2 + pad_y_bottom)
                        start_x = max(0, x1 - pad_x)
                        end_x = min(width, x2 + pad_x)
                        
                        # 1. Clean Crop for Display (No bounding box)
                        # Use the cached clean frame from the exact moment of detection to prevent misalignment
                        crop_source = cached_clean_frame if cached_clean_frame is not None else frame
                        snapshot_frame_crop = crop_source[start_y:end_y, start_x:end_x].copy()
                        sh, sw = snapshot_frame_crop.shape[:2]
                        if sw > 0 and sh > 0:
                            if sw > 400:
                                scale = 400 / sw
                                snapshot_frame_crop = cv2.resize(snapshot_frame_crop, (int(sw * scale), int(sh * scale)))
                        else:
                            scale = 640 / width
                            snapshot_frame_crop = cv2.resize(frame, (int(width * scale), int(height * scale)))
                            
                        # 2. Full Frame with Bounding Box for Logs
                        snapshot_frame_full = frame_copy.copy()
                        if width > 1280:
                            scale_full = 1280 / width
                            snapshot_frame_full = cv2.resize(snapshot_frame_full, (int(width * scale_full), int(height * scale_full)))
                        
                        last_snap = self.last_snapshot.get((cctv_id, person_name), 0)
                        is_new_session = (current_time - last_snap > 1800)
                        
                        if is_new_session:
                            self.session_uuids[(cctv_id, person_name)] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            self.last_snapshot[(cctv_id, person_name)] = current_time
                            
                        session_uuid = self.session_uuids.get((cctv_id, person_name), datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                        
                        folder_name = re.sub(r'[\\/*?:"<>|]', "", person_name).strip()
                        user_snapshot_dir = os.path.join(base_snapshot_dir, folder_name)
                        os.makedirs(user_snapshot_dir, exist_ok=True)
                        
                        last_filename = f"{session_uuid}_last.jpg"
                        last_crop_filename = f"{session_uuid}_last_crop.jpg"
                        
                        # Save both full and crop
                        cv2.imwrite(os.path.join(user_snapshot_dir, last_filename), snapshot_frame_full, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        cv2.imwrite(os.path.join(user_snapshot_dir, last_crop_filename), snapshot_frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        first_path_db = None
                        if is_new_session:
                            first_filename = f"{session_uuid}_first.jpg"
                            first_crop_filename = f"{session_uuid}_first_crop.jpg"
                            cv2.imwrite(os.path.join(user_snapshot_dir, first_filename), snapshot_frame_full, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            cv2.imwrite(os.path.join(user_snapshot_dir, first_crop_filename), snapshot_frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            first_path_db = f"{folder_name}/{first_filename}"
                            
                        last_path_db = f"{folder_name}/{last_filename}"
                        crop_path_db = f"{folder_name}/{last_crop_filename}"
                        
                        from app.config.cctv_config import get_cctv_by_id
                        cctv_info = get_cctv_by_id(cctv_id)
                        cctv_name_db = cctv_info['name'] if cctv_info else None
                        
                        repositories.logs.create_detection_log(
                            db, cctv_id, person_name, confidence, 
                            snapshot_path=first_path_db, 
                            crop_snapshot_path=crop_path_db,
                            status="present",
                            cctv_name=cctv_name_db
                        )
                        self.last_logged[(cctv_id, person_name)] = current_time
                        self.last_update_time = current_time

                # Resize frame & lower JPEG quality to reduce stream bitrate
                stream_frame = cv2.resize(frame_copy, (640, int(640 * frame_copy.shape[0] / frame_copy.shape[1])))
                success, encoded_image = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
                if success:
                    jpeg_bytes = encoded_image.tobytes()
                    base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
                    self.latest_frames[cctv_id] = base64_image
                
                time.sleep(0.01)
                
        finally:
            stream_obj.release()
            db.close()
BackgroundMonitor = BackgroundMonitorManager()
