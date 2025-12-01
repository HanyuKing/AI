from fastapi import APIRouter, HTTPException
from server.schemas.requests import (
    JsonFormatRequest, SqlFormatRequest, HashRequest, 
    Base64Request, UuidRequest, UrlEncodeRequest, PasswordGenerateRequest,
    BaseConvertRequest
)
from server.schemas.responses import BaseResponse
from server.services.logic_service import LogicService

router = APIRouter()

@router.post("/json/format", response_model=BaseResponse)
async def format_json(request: JsonFormatRequest):
    result = LogicService.format_json(request.content, request.indent, request.sort_keys)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["result"])
    return BaseResponse(data=result["result"])

@router.post("/json/to-yaml", response_model=BaseResponse)
async def json_to_yaml(request: JsonFormatRequest):
    result = LogicService.json_to_yaml(request.content)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["result"])
    return BaseResponse(data=result["result"])

@router.post("/sql/format", response_model=BaseResponse)
async def format_sql(request: SqlFormatRequest):
    formatted = LogicService.format_sql(request.sql)
    return BaseResponse(data=formatted)

@router.post("/hash", response_model=BaseResponse)
async def calculate_hash(request: HashRequest):
    try:
        result = LogicService.calculate_hash(request.text, request.algorithm)
        return BaseResponse(data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/base64", response_model=BaseResponse)
async def base64_process(request: Base64Request):
    result = LogicService.base64_process(request.text, request.action)
    return BaseResponse(data=result)

@router.post("/uuid", response_model=BaseResponse)
async def generate_uuid(request: UuidRequest):
    result = LogicService.generate_uuid(request.count, request.uppercase, request.hyphens)
    return BaseResponse(data=result)

@router.post("/url", response_model=BaseResponse)
async def url_process(request: UrlEncodeRequest):
    result = LogicService.url_process(request.text, request.action)
    return BaseResponse(data=result)

@router.post("/password", response_model=BaseResponse)
async def generate_password(request: PasswordGenerateRequest):
    result = LogicService.generate_password(
        request.length, 
        request.include_uppercase, 
        request.include_lowercase,
        request.include_digits, 
        request.include_symbols
    )
    return BaseResponse(data=result)

@router.post("/base-convert", response_model=BaseResponse)
async def base_convert(request: BaseConvertRequest):
    result = LogicService.convert_base(request.value, request.from_base, request.to_base)
    return BaseResponse(data=result)
