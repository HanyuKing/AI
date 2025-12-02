from fastapi import APIRouter, HTTPException
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

