from fastapi import APIRouter, HTTPException, Response
from server.schemas.feedback import FeedbackRequest
from server.services.feedback_service import feedback_service

router = APIRouter()

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        feedback_id = await feedback_service.submit_feedback(
            content=request.content,
            contact=request.contact,
            type=request.type
        )
        return {"status": "success", "id": feedback_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    """
    Generate sitemap.xml for SEO
    """
    base_url = "https://mediatoolbox.com"
    urls = [
        "/",
        "/tools/id-photo",
        "/tools/pdf-compress",
        "/tools/image-convert",
        "/tools/json",
        "/tools/base64",
        "/tools/uuid",
        "/tools/timestamp",
        "/tools/qrcode",
        "/tools/image-edit",
        "/tools/base-convert",
        "/tools/url-encoder",
        "/tools/password",
        "/tools/diff",
        "/tools/svg-converter",
        "/tools/pixel-converter",
        "/feedback"
    ]
    
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_content += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    
    for url in urls:
        xml_content += '  <url>\n'
        xml_content += f'    <loc>{base_url}{url}</loc>\n'
        xml_content += '    <changefreq>weekly</changefreq>\n'
        xml_content += '    <priority>0.8</priority>\n'
        xml_content += '  </url>\n'
        
    xml_content += '</urlset>'
    
    return Response(content=xml_content, media_type="application/xml")

@router.get("/robots.txt", include_in_schema=False)
async def robots():
    """
    Generate robots.txt for SEO
    """
    content = """User-agent: *
Allow: /
Disallow: /api/
Disallow: /static/private/

Sitemap: https://mediatoolbox.com/sitemap.xml
"""
    return Response(content=content, media_type="text/plain")

