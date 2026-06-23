# Konfigurasi Hardcoded CCTV

CCTVS = [
    {
        'id': 1, 
        'name': 'WebCam', 
        'rtsp_url': '0', 
        'status': False
    }
]

def get_all_cctvs():
    return CCTVS

def get_cctv_by_id(cctv_id: int):
    for cctv in CCTVS:
        if cctv['id'] == cctv_id:
            return cctv
    return None
