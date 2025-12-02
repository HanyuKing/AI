from pydantic import BaseModel
from typing import Optional

class FeedbackRequest(BaseModel):
    content: str
    contact: Optional[str] = None
    type: str = "suggestion"  # suggestion, bug, praise

