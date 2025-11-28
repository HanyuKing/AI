from fastapi import APIRouter, HTTPException, Response
from server.schemas.requests import (
    TimestampRequest, BaseConvertRequest, QrCodeRequest
)
from server.schemas.responses import BaseResponse
from server.services.logic_service import LogicService

router = APIRouter()

@router.post("/time/convert", response_model=BaseResponse)
async def convert_time(request: TimestampRequest):
    result = LogicService.convert_timestamp(request.timestamp)
    return BaseResponse(data=result)

@router.post("/math/base-convert", response_model=BaseResponse)
async def convert_base(request: BaseConvertRequest):
    result = LogicService.convert_base(request.value, request.from_base, request.to_base)
    return BaseResponse(data=result)

@router.post("/qrcode")
async def generate_qrcode(request: QrCodeRequest):
    img_bytes = LogicService.generate_qrcode(request.text, request.fill_color, request.back_color)
    return Response(content=img_bytes, media_type="image/png")

