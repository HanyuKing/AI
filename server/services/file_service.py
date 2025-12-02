import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Callable, Awaitable

# Add project root to sys.path to import existing utils
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from file_utils.file_tool import FileTool

class FileService:
    @staticmethod
    async def compress_pdf(
        input_path: str, 
        output_path: str, 
        ratio: float,
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> None:
        await FileTool.compress_pdf(input_path, output_path, ratio, progress_callback)

    @staticmethod
    def convert_format(input_path: str, output_path: str, target_format: Optional[str] = None) -> None:
        FileTool.convert_image_format(input_path, output_path, target_format)

    @staticmethod
    def compress_image(input_path: str, output_path: str, ratio: float = None, max_size_kb: int = None) -> None:
        if ratio:
            FileTool.compress_image_by_ratio(input_path, output_path, ratio)
        elif max_size_kb:
            FileTool.compress_image_to_size(input_path, output_path, max_size_kb)
        else:
            # If no params, just copy
            shutil.copy2(input_path, output_path)
