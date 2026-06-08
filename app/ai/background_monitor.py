import threading
import time
import cv2
import base64
from app.database.database import SessionLocal
from app import repositories
from app.ai.engine import predict_frame
from app.ai.tracker import FaceTracker

class CameraStreamReader:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        self.ret = False
        self.frame = None
        self.running = True
        
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
                else:
                    time.sleep(1)
            else:
                time.sleep(1)

    def read(self):
        return self.ret, self.frame
        
    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

class BackgroundMonitorManager:
    def __init__(self):
        self.active_threads = {}
        self.stop_events = {}
        self.latest_frames = {}  # cctv_id -> base64 string

    def get_latest_frame(self, cctv_id):
        return self.latest_frames.get(cctv_id)

    def start_camera(self, cctv_id, rtsp_url):
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
        if cctv_id in self.stop_events:
            self.stop_events[cctv_id].set()
            if cctv_id in self.active_threads:
                self.active_threads[cctv_id].join(timeout=2.0)
                del self.active_threads[cctv_id]
            del self.stop_events[cctv_id]
            print(f"Background monitoring stopped for CCTV {cctv_id}")

    def start_all(self):
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
        db = SessionLocal()
        stream_obj = CameraStreamReader(rtsp_url)
        tracker = FaceTracker()
        
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

                # Prepare data for tracker
                rects = []
                current_names = []
                current_user_ids = []
                current_confidences = []
                for face in faces_detected:
                    if face.get("box"):
                        x1, y1, x2, y2 = face.get("box")
                        rects.append((x1, y1, x2, y2))
                        current_names.append(face.get("name", "Unknown"))
                        current_user_ids.append(face.get("user_id"))
                        current_confidences.append(face.get("confidence", 0.0))

                smoothed_names, smoothed_user_ids, smoothed_confidences = tracker.update(
                    rects, current_names, current_user_ids, current_confidences
                )

                for i, rect in enumerate(rects):
                    user_id = smoothed_user_ids[i]
                    confidence = smoothed_confidences[i]
                    name_label = smoothed_names[i]
                    x1, y1, x2, y2 = rect
                    
                    lecture = None
                    if user_id:
                        lecture = repositories.lectures.get_lecture_by_id(db, user_id)
                    elif name_label.lower() != "unknown":
                        lecture = repositories.lectures.get_lecture_by_nis(db, name_label)
                        
                    if lecture:
                        # Log to DB
                        repositories.lectures.create_detection_log(db, cctv_id, lecture.id, confidence)
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
                
                # Small sleep to prevent 100% CPU
                time.sleep(0.03)
                
        finally:
            stream_obj.release()
            db.close()

BackgroundMonitor = BackgroundMonitorManager()
