"""
Utils package for server
Contains utility classes and functions
"""

# 只导入规格数据（不依赖重量级库）
from server.utils.id_photo_specs import (
    ID_PHOTO_SPECS,
    BG_COLOR_PRESETS,
    get_spec_by_id,
    get_all_specs,
    search_specs
)

# IdPhotoProcessor 和 ID_PHOTO_SIZES 需要延迟导入（避免启动时加载 rembg）
# 使用时请直接导入：from server.utils.id_photo import IdPhotoProcessor

__all__ = [
    # 规格数据（可立即导入）
    'ID_PHOTO_SPECS',
    'BG_COLOR_PRESETS',
    'get_spec_by_id',
    'get_all_specs',
    'search_specs',
    # 处理器（需延迟导入，不在 __init__ 中导入）
    # 'IdPhotoProcessor',  # 需要时使用: from server.utils.id_photo import IdPhotoProcessor
    # 'ID_PHOTO_SIZES',
]
