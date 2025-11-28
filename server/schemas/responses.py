from typing import Any, Optional
from pydantic import BaseModel

class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Success"
    data: Any = None

class FileResponseInfo(BaseModel):
    filename: str
    file_url: str
    file_size: int

