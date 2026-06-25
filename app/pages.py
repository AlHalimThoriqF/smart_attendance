from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.config.cctv_config import get_all_cctvs

router = APIRouter(tags=["pages"])
# Get the absolute path to the templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=templates_dir)


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    cctvs = get_all_cctvs()
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"active_page": "dashboard", "cctvs": cctvs})
    
@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    return templates.TemplateResponse(request=request, name="logs.html", context={"active_page": "logs"})

@router.get("/display", response_class=HTMLResponse)
async def display_page(request: Request):
    return templates.TemplateResponse(request=request, name="display.html", context={"active_page": "display"})

@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    cctvs = get_all_cctvs()
    return templates.TemplateResponse(request=request, name="upload.html", context={"active_page": "upload", "cctvs": cctvs})
