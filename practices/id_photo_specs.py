"""
向后兼容层：practices/id_photo_specs.py
仅用于命令行工具，实际数据已移至 server/utils/id_photo_specs.py
"""

import sys
import os

# 添加项目根目录到 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 从实际位置导入所有内容
from server.utils.id_photo_specs import (
    ID_PHOTO_SPECS,
    BG_COLOR_PRESETS,
    get_all_specs,
    get_spec_by_id,
    search_specs
)

__all__ = [
    'ID_PHOTO_SPECS',
    'BG_COLOR_PRESETS',
    'get_all_specs',
    'get_spec_by_id',
    'search_specs',
]

# 为了向后兼容测试代码
if __name__ == "__main__":
    print("所有分类:")
    for cat_id, cat_data in ID_PHOTO_SPECS.items():
        print(f"  {cat_data['icon']} {cat_data['category_name']}: {len(cat_data['specs'])} 种规格")
    
    print("\n搜索测试 - '公务员':")
    results = search_specs("公务员")
    for r in results:
        print(f"  - {r['name']}: {r['size_mm']} mm")
