from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from server.core.config import settings
from server.core.exceptions import global_exception_handler
from server.api import media, dev_tools, utils_tools, general, websocket

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
app.include_router(general.router, prefix="/api/general", tags=["General"])
app.include_router(websocket.router, tags=["WebSocket"])

# --- Frontend Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})

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

@app.get("/tools/uuid", response_class=HTMLResponse)
async def tool_uuid(request: Request):
    return templates.TemplateResponse("tools/uuid.html", {"request": request})

@app.get("/tools/timestamp", response_class=HTMLResponse)
async def tool_timestamp(request: Request):
    return templates.TemplateResponse("tools/timestamp.html", {"request": request})

@app.get("/tools/qrcode", response_class=HTMLResponse)
async def tool_qrcode(request: Request):
    return templates.TemplateResponse("tools/qrcode.html", {"request": request})

@app.get("/tools/image-edit", response_class=HTMLResponse)
async def tool_image_edit(request: Request):
    return templates.TemplateResponse("tools/image_edit.html", {"request": request})

@app.get("/tools/base-convert", response_class=HTMLResponse)
async def tool_base_convert(request: Request):
    return templates.TemplateResponse("tools/base_convert.html", {"request": request})

@app.get("/tools/url-encoder", response_class=HTMLResponse)
async def tool_url_encoder(request: Request):
    return templates.TemplateResponse("tools/url_encoder.html", {"request": request})

@app.get("/tools/password", response_class=HTMLResponse)
async def tool_password(request: Request):
    return templates.TemplateResponse("tools/password.html", {"request": request})

@app.get("/tools/diff", response_class=HTMLResponse)
async def tool_diff(request: Request):
    return templates.TemplateResponse("tools/diff.html", {"request": request})

@app.get("/tools/svg-converter", response_class=HTMLResponse)
async def tool_svg_converter(request: Request):
    return templates.TemplateResponse("tools/svg_converter.html", {"request": request})

@app.get("/tools/id-photo", response_class=HTMLResponse)
async def tool_id_photo(request: Request):
    return templates.TemplateResponse("tools/id_photo.html", {"request": request})

@app.get("/tools/pixel-converter", response_class=HTMLResponse)
async def tool_pixel_converter(request: Request):
    return templates.TemplateResponse("tools/pixel_converter.html", {"request": request})

@app.get("/tools/base64-to-image", response_class=HTMLResponse)
async def tool_base64_to_image(request: Request):
    return templates.TemplateResponse("tools/base64_to_image.html", {"request": request})

@app.get("/tools/calligraphy", response_class=HTMLResponse)
async def tool_calligraphy(request: Request):
    return templates.TemplateResponse("tools/calligraphy.html", {"request": request})


@app.get("/sitemap.xml", response_class=HTMLResponse)
async def sitemap(request: Request):
    base_url = str(request.base_url).rstrip("/")
    urls = [
        {"loc": f"{base_url}/", "changefreq": "daily", "priority": "1.0"},
        {"loc": f"{base_url}/tools/pdf-compress", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/image-convert", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/json", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/base64", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/base64-to-image", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/calligraphy", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/uuid", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/timestamp", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/qrcode", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/image-edit", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/base-convert", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/url-encoder", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/password", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/diff", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/svg-converter", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/id-photo", "changefreq": "weekly", "priority": "0.8"},
        {"loc": f"{base_url}/tools/pixel-converter", "changefreq": "weekly", "priority": "0.8"},
    ]
    
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
"""
    for url in urls:
        xml_content += f"""    <url>
        <loc>{url['loc']}</loc>
        <changefreq>{url['changefreq']}</changefreq>
        <priority>{url['priority']}</priority>
    </url>
"""
    xml_content += "</urlset>"
    return HTMLResponse(content=xml_content, media_type="application/xml")

@app.get("/robots.txt", response_class=HTMLResponse)
async def robots(request: Request):
    base_url = str(request.base_url).rstrip("/")
    content = f"""User-agent: *
Allow: /
Sitemap: {base_url}/sitemap.xml
"""
    return HTMLResponse(content=content, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
