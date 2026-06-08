from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

router = APIRouter(tags=["pages"])
# Get the absolute path to the templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=templates_dir)

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={"active_page": "login"})

@router.get("/live", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"active_page": "dashboard"})

@router.get("/cctv", response_class=HTMLResponse)
async def cctv_page(request: Request):
    return templates.TemplateResponse(request=request, name="cctv.html", context={"active_page": "cctv"})

@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    return templates.TemplateResponse(request=request, name="logs.html", context={"active_page": "logs"})

@router.get("/", response_class=HTMLResponse)
async def display_page(request: Request):
    return templates.TemplateResponse(request=request, name="display.html", context={"active_page": "display"})
