import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "在线工具"
    VERSION: str = "1.0.0"
    
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    
    # Temporary Directory for File Processing
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    def __init__(self):
        # Ensure temp directory exists
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()

