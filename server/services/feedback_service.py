import abc
from typing import List, Dict, Any
import json
import os
import time
from pathlib import Path

class FeedbackStorage(abc.ABC):
    @abc.abstractmethod
    async def save_feedback(self, feedback_data: Dict[str, Any]) -> str:
        pass

    @abc.abstractmethod
    async def get_feedbacks(self, limit: int = 100) -> List[Dict[str, Any]]:
        pass

class FileFeedbackStorage(FeedbackStorage):
    def __init__(self, file_path: str = "data/feedbacks.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    async def save_feedback(self, feedback_data: Dict[str, Any]) -> str:
        # Simple file append implementation
        # In production, this should have file locking
        try:
            with open(self.file_path, "r+", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                
                feedback_data["id"] = str(len(data) + 1)
                feedback_data["timestamp"] = int(time.time())
                data.append(feedback_data)
                
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
                
            return feedback_data["id"]
        except Exception as e:
            print(f"Error saving feedback: {e}")
            raise e

    async def get_feedbacks(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data[-limit:]
        except FileNotFoundError:
            return []

class FeedbackService:
    def __init__(self, storage: FeedbackStorage):
        self.storage = storage

    async def submit_feedback(self, content: str, contact: str = None, type: str = "suggestion"):
        feedback = {
            "content": content,
            "contact": contact,
            "type": type
        }
        return await self.storage.save_feedback(feedback)

# Singleton instance with file storage for now
# Easy to swap with DBFeedbackStorage later
_storage = FileFeedbackStorage()
feedback_service = FeedbackService(_storage)

