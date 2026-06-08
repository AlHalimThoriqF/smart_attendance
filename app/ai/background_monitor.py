import threading
import time
import cv2
import base64
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
        
        try:
            while not stop_event.is_set():
                ret, frame = stream_obj.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
                
                frame_copy = frame.copy()
                faces_detected = []
                try:
                    faces_detected = predict_frame(frame_copy)
                except Exception as e:
                    print(f"AI prediction error on camera {cctv_id}: {e}")

                for face in faces_detected:
                    if not face.get("box"):
                        continue
                    x1, y1, x2, y2 = face.get("box")
                    user_id = face.get("user_id")
                    confidence = face.get("confidence", 0.0)
                    name_label = face.get("name", "Unknown")
                    
                    lecture = None
                    if user_id:
                        lecture = repositories.lectures.get_lecture_by_id(db, user_id)
                    elif name_label.lower() != "unknown":
                        lecture = repositories.lectures.get_lecture_by_nis(db, name_label)
                        
                    if lecture:
                        # Log to DB dengan cooldown pendek (3 detik) agar last_seen terasa realtime
                        # namun tetap mencegah spam ke database pada 30 FPS.
                        current_time = time.time()
                        last_time = self.last_logged.get((cctv_id, lecture.id), 0)
                        if current_time - last_time > 3: # 5 detik
                            repositories.lectures.create_detection_log(db, cctv_id, lecture.id, confidence)
                            self.last_logged[(cctv_id, lecture.id)] = current_time
                            self.last_update_time = current_time
                            
                        name_label = lecture.name

                    color = (0, 0, 255) if name_label.lower() == "unknown" else (0, 255, 0)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                    
                    if name_label.lower() == "unknown" or name_label == "UNKNOWN":
                        label = "Unknown"
                    else:
                        label = f"{name_label} ({confidence*100:.1f}%)" if confidence > 0 else name_label
                    
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(frame_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                success, encoded_image = cv2.imencode('.jpg', frame_copy)
                if success:
                    jpeg_bytes = encoded_image.tobytes()
                    base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
                    self.latest_frames[cctv_id] = base64_image
                
                time.sleep(0.03)
                
        finally:
            stream_obj.release()
            db.close()

BackgroundMonitor = BackgroundMonitorManager()
