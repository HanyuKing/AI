from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from server.core.config import settings
from server.core.exceptions import global_exception_handler
from server.api import media, dev_tools, utils_tools

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="A Swiss Army Knife for Developers and Office Workers"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(Exception, global_exception_handler)

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Include Routers
app.include_router(media.router, prefix="/api/media", tags=["Media"])
app.include_router(dev_tools.router, prefix="/api/dev", tags=["Dev Tools"])
app.include_router(utils_tools.router, prefix="/api/utils", tags=["Utilities"])

# --- Frontend Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/tools/pdf-compress", response_class=HTMLResponse)
async def tool_pdf_compress(request: Request):
    return templates.TemplateResponse("tools/pdf_compress.html", {"request": request})

@app.get("/tools/image-convert", response_class=HTMLResponse)
async def tool_image_convert(request: Request):
    return templates.TemplateResponse("tools/image_convert.html", {"request": request})

@app.get("/tools/json", response_class=HTMLResponse)
async def tool_json(request: Request):
    return templates.TemplateResponse("tools/json.html", {"request": request})

@app.get("/tools/base64", response_class=HTMLResponse)
async def tool_base64(request: Request):
    return templates.TemplateResponse("tools/base64.html", {"request": request})

# Add more routes as I create the templates...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=80, reload=True)
