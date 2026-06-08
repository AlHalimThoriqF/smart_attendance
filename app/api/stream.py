import asyncio
import base64
import os
import cv2
import threading
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ai.recognition import predict_frame
from app.database.database import SessionLocal
from app import repositories

class CameraStream:
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
                # Read as fast as possible to prevent buffer overflow and h264 macroblock corruption
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

from app.ai.background_monitor import BackgroundMonitor

router = APIRouter(prefix="/ws", tags=["stream"])

@router.websocket("/display")
async def websocket_display(websocket: WebSocket):
    await websocket.accept()
    last_notified = BackgroundMonitor.last_update_time
    try:
        while True:
            if BackgroundMonitor.last_update_time > last_notified:
                last_notified = BackgroundMonitor.last_update_time
                await websocket.send_json({"event": "refresh"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

@router.websocket("/stream/{cctv_id}")
async def websocket_stream(websocket: WebSocket, cctv_id: int):
    await websocket.accept()
    
    # If the camera isn't running in the background for some reason, try to start it
    # We should fetch its RTSP url to start it if needed.
    db = SessionLocal()
    camera = repositories.cctv.get_cctv(db, cctv_id)
    if camera and camera.status:
        rtsp = camera.rtsp_url
        if rtsp.isdigit():
            rtsp = int(rtsp)
        BackgroundMonitor.start_camera(cctv_id, rtsp)
    db.close()

    try:
        while True:
            # Simply poll the latest ready frame from the background monitor
            frame_b64 = BackgroundMonitor.get_latest_frame(cctv_id)
            if frame_b64:
                try:
                    await websocket.send_json({
                        "base64_image": frame_b64
                    })
                except WebSocketDisconnect:
                    break
            
            # Send frames at roughly 20-30 FPS
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print(f"Websocket disconnected for CCTV ID {cctv_id}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
