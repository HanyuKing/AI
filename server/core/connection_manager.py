import asyncio
from typing import Dict, List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_progress: Dict[str, int] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
        if task_id in self.task_progress:
            del self.task_progress[task_id]

    async def update_progress(self, task_id: str, progress: int, message: str = ""):
        self.task_progress[task_id] = progress
        if task_id in self.active_connections:
            websocket = self.active_connections[task_id]
            try:
                await websocket.send_json({"progress": progress, "message": message})
            except Exception:
                # Connection might be closed
                pass

manager = ConnectionManager()

