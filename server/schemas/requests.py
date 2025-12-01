from pydantic import BaseModel, Field
from typing import Optional

# --- JSON / Data Tools ---
class JsonFormatRequest(BaseModel):
    content: str
    indent: int = 4
    sort_keys: bool = False

class SqlFormatRequest(BaseModel):
    sql: str

class HashRequest(BaseModel):
    text: str
    algorithm: str = "md5"  # md5, sha1, sha256, sha512

class Base64Request(BaseModel):
    text: str
    action: str = "encode"  # encode, decode

class UrlEncodeRequest(BaseModel):
    text: str
    action: str = "encode" # encode, decode

# --- Time / Date Tools ---
class TimestampRequest(BaseModel):
    timestamp: Optional[float] = None
    format: str = "%Y-%m-%d %H:%M:%S"

# --- Math / Number Tools ---
class BaseConvertRequest(BaseModel):
    value: str
    from_base: int
    to_base: int

# --- Generator Tools ---
class QrCodeRequest(BaseModel):
    text: str
    fill_color: str = "black"
    back_color: str = "white"

class UuidRequest(BaseModel):
    count: int = 1
    uppercase: bool = False
    hyphens: bool = True

class PasswordGenerateRequest(BaseModel):
    length: int = 16
    include_uppercase: bool = True
    include_lowercase: bool = True
    include_digits: bool = True
    include_symbols: bool = True
