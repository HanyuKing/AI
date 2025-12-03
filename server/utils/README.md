# Server Utils

服务器端工具类集合。

## 📁 目录结构

```
server/utils/
├── __init__.py                 # 包初始化（仅导出规格数据）
├── id_photo_specs.py           # 证件照规格数据库（33种规格）
├── id_photo.py                 # 证件照处理核心类
└── README.md                   # 本文档
```

## 📦 模块说明

### id_photo_specs.py

**证件照规格数据库**

包含33种证件照规格，按8大场景分类：
- 📏 常用标准（5种）
- 🪪 身份证件（3种）
- ✈️ 出入境证件（5种）
- 🚗 驾驶证件（2种）
- 📝 考试报名（7种）
- 🎓 学历学位（3种）
- 💼 职业资格（3种）
- 🏥 社保医疗（2种）
- 📋 其他用途（3种）

**导出内容**:
- `ID_PHOTO_SPECS` - 完整的规格数据字典
- `BG_COLOR_PRESETS` - 背景颜色预设
- `get_all_specs()` - 获取所有规格（扁平化）
- `get_spec_by_id(spec_id)` - 根据ID获取规格
- `search_specs(keyword)` - 搜索规格

**使用示例**:
```python
from server.utils.id_photo_specs import ID_PHOTO_SPECS, get_spec_by_id, search_specs

# 获取特定规格
spec = get_spec_by_id('driving_license')
print(f"{spec['name']}: {spec['size_mm']} mm")

# 搜索规格
results = search_specs('公务员')
for r in results:
    print(f"{r['name']}: {r['size_mm']} mm")
```

### id_photo.py

**证件照处理核心类**

提供完整的证件照处理功能：
- AI人像抠图（使用 BiRefNet-portrait 模型）
- 背景替换
- 智能美颜（磨皮、美白）
- 精准尺寸调整
- 高清输出（300 DPI）

**导出内容**:
- `IdPhotoProcessor` - 核心处理类
- `ID_PHOTO_SIZES` - 向后兼容的尺寸字典

**使用示例**:
```python
from server.utils.id_photo import IdPhotoProcessor

# 创建处理器
processor = IdPhotoProcessor('input.jpg')

# 生成证件照
result = processor.generate_id_photo(
    spec_id='driving_license',
    bg_color_name='white',
    use_beautify=True
)

# 保存
result.save('output.jpg', quality=95, dpi=(300, 300))
```

**注意**: 
- 导入此模块会加载 `rembg` 库（约1GB模型文件）
- 在 Python 3.13 下可能遇到兼容性问题，建议使用 Python 3.11/3.12
- 如需在 Python 3.13 下使用，请先设置环境变量：`os.environ['NUMBA_CACHE_DIR'] = '/tmp'`

## 🚀 快速开始

### 方式1: 导入规格数据（轻量）

```python
from server.utils import ID_PHOTO_SPECS, get_spec_by_id, search_specs

# 获取所有分类
for category_id, category in ID_PHOTO_SPECS.items():
    print(f"{category['category_name']}: {len(category['specs'])} 种")

# 查询特定规格
spec = get_spec_by_id('driving_license')

# 搜索规格
results = search_specs('公务员')
```

### 方式2: 使用处理器（重量级）

```python
# 延迟导入（避免启动时加载重量级库）
from server.utils.id_photo import IdPhotoProcessor

processor = IdPhotoProcessor('photo.jpg')
result = processor.generate_id_photo(
    spec_id='driving_license',
    bg_color_name='white'
)
result.save('output.jpg')
```

## 📊 与 practices 目录的关系

### 代码组织

**server/utils/** - 生产代码
- 实际的工具类和数据定义
- 被 server 服务层调用
- 优化的导入和性能

**practices/** - 兼容层 + 命令行工具
- `practices/id_photo_utils.py` - 转发到 `server.utils.id_photo`
- `practices/id_photo_specs.py` - 转发到 `server.utils.id_photo_specs`
- `practices/id_photo_maker.py` - 命令行交互工具
- `practices/id_photo_quick.py` - 命令行快捷工具
- `practices/show_all_specs.py` - 规格查看工具

### 导入关系

```
practices/
  id_photo_utils.py     ─┐
  id_photo_specs.py     ─┼─→ server/utils/
  id_photo_maker.py     ─┤     id_photo.py
  id_photo_quick.py     ─┤     id_photo_specs.py
  show_all_specs.py     ─┘
                              ↓
                         server/services/
                           id_photo_service.py
                              ↓
                         server/api/
                           media.py
```

## 🔧 开发指南

### 添加新规格

编辑 `id_photo_specs.py`，在对应分类下添加：

```python
"new_spec_id": {
    "name": "规格名称",
    "size_mm": (宽, 高),
    "size_px": (像素宽, 像素高),  # @300 DPI
    "common_bg": ["white", "blue"],
    "description": "规格描述",
    "usage": "使用场景",
    "note": "注意事项（可选）"
}
```

### 修改处理算法

编辑 `id_photo.py` 中的相关方法：
- `_post_process()` - 抠图后处理
- `remove_background()` - 背景移除
- `beautify()` - 美颜处理
- `generate_id_photo()` - 主流程

## ⚠️ 注意事项

### 性能考虑

1. **规格数据** - 可以随时导入，轻量级
2. **处理器类** - 延迟导入，避免启动时加载 rembg

```python
# ❌ 不推荐：直接从 __init__ 导入
from server.utils import IdPhotoProcessor  # 会加载 rembg

# ✅ 推荐：延迟导入
from server.utils.id_photo import IdPhotoProcessor  # 仅在需要时导入
```

### Python 版本兼容性

- **Python 3.11/3.12** - 完全支持，推荐使用
- **Python 3.13** - 需要设置环境变量解决 pymatting 兼容性
  ```python
  import os
  os.environ['NUMBA_CACHE_DIR'] = '/tmp'
  ```

### 依赖库

核心依赖：
- `opencv-python` - 图像处理
- `numpy` - 数值计算
- `Pillow` - 图像IO
- `rembg` - AI抠图（约1GB模型）

## 📚 相关文档

- [快速入门指南](../../practices/快速入门.md)
- [完整使用手册](../../practices/ID_PHOTO_README.md)
- [API文档](../services/id_photo_service.py)

## 🔄 版本历史

### v2.0 (2025-12-03)
- ✅ 重构代码到 `server/utils/`
- ✅ 33种证件照规格支持
- ✅ 场景分类和搜索功能
- ✅ 向后兼容 practices 目录
- ✅ 优化导入性能

### v1.0
- 基础证件照处理功能
- 7种常用尺寸
