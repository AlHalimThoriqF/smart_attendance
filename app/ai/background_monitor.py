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
        # Inisialisasi koneksi ke stream RTSP dan mulai thread pembacaan frame.
        self.kamera = cv2.VideoCapture(rtsp_url)
        self.ret = False
        self.frame = None
        self.running = True
        
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
                else:
                    time.sleep(1)
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
        self.last_logged = {} # Menyimpan waktu terakhir log untuk tiap (cctv_id, lecture_id)
        self.last_snapshot = {} # Menyimpan kapan terakhir snapshot fisik diambil
        self.session_uuids = {} # Menyimpan UUID sesi untuk tiap (cctv_id, lecture_id)
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
        # Memulai monitoring background untuk semua kamera yang berstatus aktif di database.
        db = SessionLocal()
        try:
            cameras = repositories.cctv.get_all_cctvs(db)
            for camera in cameras:
                if camera.status:
                    rtsp = camera.rtsp_url
                    if rtsp.isdigit():
                        rtsp = int(rtsp)
                    self.start_camera(camera.id, rtsp)
        finally:
            db.close()

    def _camera_worker(self, cctv_id, rtsp_url, stop_event):
        # Worker utama untuk membaca stream, melakukan deteksi wajah, tracking, dan logging.
        db = SessionLocal()
        stream_obj = CameraStreamReader(rtsp_url)
        
        last_ai_time = 0
        cached_valid_detections = []
        cached_faces_detected = []
        
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
                months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
                day_name = days[now.weekday()]
                month_name = months[now.month - 1]
                timestamp_str = f"{day_name}, {now.day:02d} {month_name} {now.year} - {now.strftime('%H:%M:%S')} WIB"
                
                text_x = 15
                text_y = 30
                cv2.putText(frame_copy, timestamp_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame_copy, timestamp_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # AI Inference (Throttled to 2 FPS)
                if current_time - last_ai_time >= 0.5:
                    try:
                        cached_faces_detected = predict_frame(frame_copy)
                    except Exception as e:
                        print(f"AI prediction error on camera {cctv_id}: {e}")
                    
                    last_ai_time = current_time
                    valid_detections = []
                    
                    for face in cached_faces_detected:
                        if not face.get("box"):
                            continue
                        user_id = face.get("user_id")
                        name_label = face.get("name", "Unknown")
                        confidence = face.get("confidence", 0.0)
                        
                        lecture = None
                        if user_id:
                            lecture = repositories.lectures.get_lecture_by_id(db, user_id)
                        elif name_label.lower() != "unknown":
                            lecture = repositories.lectures.get_lecture_by_name(db, name_label)
                            
                        if lecture:
                            name_label = lecture.name
                            valid_detections.append({
                                "lecture": lecture,
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
                        label = f"{name_label} ({confidence*100:.1f}%)" if confidence > 0 else name_label
                    
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(frame_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Database Logging
                needs_logging = []
                for det in cached_valid_detections:
                    lecture = det["lecture"]
                    last_time = self.last_logged.get((cctv_id, lecture.id), 0)
                    if current_time - last_time > 3:
                        needs_logging.append(det)
                        
                if needs_logging:
                    height, width = frame_copy.shape[:2]
                    scale = 640 / width
                    new_width, new_height = int(width * scale), int(height * scale)
                    snapshot_frame = cv2.resize(frame_copy, (new_width, new_height))
                    
                    base_snapshot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "snapshots")
                    
                    for det in needs_logging:
                        lecture = det["lecture"]
                        confidence = det["confidence"]
                        
                        last_snap = self.last_snapshot.get((cctv_id, lecture.id), 0)
                        is_new_session = (current_time - last_snap > 1800)
                        
                        if is_new_session:
                            self.session_uuids[(cctv_id, lecture.id)] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            self.last_snapshot[(cctv_id, lecture.id)] = current_time
                            
                        session_uuid = self.session_uuids.get((cctv_id, lecture.id), datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                        
                        folder_name = re.sub(r'[\\/*?:"<>|]', "", lecture.name).strip()
                        user_snapshot_dir = os.path.join(base_snapshot_dir, folder_name)
                        os.makedirs(user_snapshot_dir, exist_ok=True)
                        
                        last_filename = f"{session_uuid}_last.jpg"
                        last_path = os.path.join(user_snapshot_dir, last_filename)
                        cv2.imwrite(last_path, snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        
                        first_path_db = None
                        if is_new_session:
                            first_filename = f"{session_uuid}_first.jpg"
                            first_path = os.path.join(user_snapshot_dir, first_filename)
                            cv2.imwrite(first_path, snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            first_path_db = f"{folder_name}/{first_filename}"
                            
                        last_path_db = f"{folder_name}/{last_filename}"
                        
                        repositories.lectures.create_detection_log(
                            db, cctv_id, lecture.id, confidence, 
                            snapshot_path=first_path_db, 
                            last_snapshot_path=last_path_db
                        )
                        self.last_logged[(cctv_id, lecture.id)] = current_time
                        self.last_update_time = current_time

                success, encoded_image = cv2.imencode('.jpg', frame_copy)
                if success:
                    jpeg_bytes = encoded_image.tobytes()
                    base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
                    self.latest_frames[cctv_id] = base64_image
                
                time.sleep(0.01)
                
        finally:
            stream_obj.release()
            db.close()

BackgroundMonitor = BackgroundMonitorManager()
