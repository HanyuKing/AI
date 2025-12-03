"""
向后兼容层：practices/id_photo_utils.py
仅用于命令行工具，实际代码已移至 server/utils/id_photo.py

注意：导入此模块会加载重量级的 rembg 库
"""

import sys
import os

# 添加项目根目录到 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 设置环境变量解决 Python 3.13 兼容性问题
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp')

# 导入实际的工具类（会加载 rembg）
from server.utils.id_photo import IdPhotoProcessor, ID_PHOTO_SIZES
from server.utils.id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS, get_spec_by_id, search_specs

# 导出所有符号，保持向后兼容
__all__ = ['IdPhotoProcessor', 'ID_PHOTO_SIZES', 'ID_PHOTO_SPECS', 'BG_COLOR_PRESETS', 'get_spec_by_id', 'search_specs']
