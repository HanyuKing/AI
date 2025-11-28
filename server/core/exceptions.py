from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal Server Error: {str(exc)}", "success": False}
    )

