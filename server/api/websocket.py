from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from server.core.connection_manager import manager

router = APIRouter()

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await manager.connect(task_id, websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id)

