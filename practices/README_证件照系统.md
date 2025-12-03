# 📸 智能证件照制作系统

## 🎯 一句话介绍

**AI驱动的智能证件照制作系统**，支持33种证件照规格，按场景分类，一键美颜换底，300 DPI高清输出。

## ⚡ 最新更新 (2025-12-03)

- ✅ **修复了模块导入错误** - 支持从 Server 正常导入
- ✅ **优化了启动性能** - 延迟加载重量级库，Server启动更快
- ✅ **新增查询API** - 支持规格查询、搜索、背景颜色获取
- ✅ **完全向后兼容** - 旧代码无需修改
- ⚠️ **Python 3.13 注意事项** - 建议使用 Python 3.11/3.12，或查看[已知问题和解决方案.md](./已知问题和解决方案.md)

---

## 🚀 快速开始

### 新手用户（5分钟上手）

```bash
# 1. 查看支持的证件类型
python show_all_specs.py summary

# 2. 交互式制作证件照
python id_photo_maker.py
```

📖 **详细教程**: [快速入门.md](./快速入门.md)

### 熟练用户（一行命令）

```bash
# 驾驶证照片（白底）
python id_photo_quick.py photo.jpg driving_license white

# 公务员考试（蓝底）
python id_photo_quick.py photo.jpg civil_servant blue

# 查看所有规格
python id_photo_quick.py --list-specs
```

### 开发者（编程集成）

```python
from id_photo_utils import IdPhotoProcessor

processor = IdPhotoProcessor("photo.jpg")
result = processor.generate_id_photo(
    spec_id="driving_license",
    bg_color_name="white"
)
result.save("output.jpg", quality=95, dpi=(300, 300))
```

📖 **代码示例**: [example_usage.py](./example_usage.py)

---

## 📂 文件导航

### 📱 使用工具

| 文件 | 用途 | 适合人群 |
|------|------|---------|
| **id_photo_maker.py** | 交互式制作工具 | ⭐ 新手推荐 |
| **id_photo_quick.py** | 快捷命令行工具 | ⭐ 熟手推荐 |
| **show_all_specs.py** | 查看所有规格 | 查询信息 |
| **example_usage.py** | 编程示例代码 | ⭐ 开发者推荐 |

### 📚 文档说明

| 文件 | 内容 | 推荐阅读 |
|------|------|---------|
| **快速入门.md** | 5分钟上手指南 | ⭐⭐⭐ 必读 |
| **ID_PHOTO_README.md** | 完整使用说明 | ⭐⭐ 深入了解 |
| **证件照系统优化报告.md** | 系统优化详情 | ⭐ 技术细节 |

### 🔧 核心代码

| 文件 | 功能 | 开发者参考 |
|------|------|-----------|
| **id_photo_specs.py** | 规格数据库（33种） | 添加新规格 |
| **id_photo_utils.py** | 核心处理类 | API文档 |

---

## 📋 支持的证件类型

### 🔥 高频场景

| 证件 | 规格ID | 尺寸 | 背景 |
|------|--------|------|------|
| 驾驶证 | `driving_license` | 22×32 mm | 白色 |
| 公务员考试 | `civil_servant` | 35×45 mm | 白/蓝 |
| 护照 | `passport` | 33×48 mm | 白色 |
| 身份证 | `id_card` | 26×32 mm | 白色 |
| 英语四六级 | `cet` | 25×35 mm | 蓝色 |
| 一寸照 | `1inch` | 25×35 mm | 白/蓝/红 |
| 二寸照 | `2inch` | 35×49 mm | 白/蓝/红 |

### 📊 完整列表

**总计33种规格**，涵盖8大分类：

1. 📏 **常用标准** (5种) - 一寸、二寸等基础规格
2. 🪪 **身份证件** (3种) - 身份证、户口本等
3. ✈️ **出入境证件** (5种) - 护照、签证、港澳台通行证
4. 🚗 **驾驶证件** (2种) - 驾驶证、从业资格证
5. 📝 **考试报名** (7种) - 公考、四六级、高考、考研、教资等
6. 🎓 **学历学位** (3种) - 学生证、毕业照、学历认证
7. 💼 **职业资格** (3种) - 健康证、工作证、职业资格证
8. 🏥 **社保医疗** (2种) - 社保卡、医保卡
9. 📋 **其他用途** (3种) - 简历照、结婚证、居住证

**查看详情**:
```bash
python show_all_specs.py
```

---

## ✨ 核心功能

### 1. 智能抠图
- SOTA人像分割模型 BiRefNet
- 发丝级边缘处理
- 自动背景移除

### 2. 背景换色
- 白色、蓝色、红色、灰色
- 透明背景（PNG）
- 智能推荐颜色

### 3. 智能美颜
- 自动磨皮美白
- 效果自然不夸张
- 可选开启/关闭

### 4. 精准尺寸
- 33种标准规格
- 300 DPI高清输出
- 智能裁剪适配

### 5. 场景分类
- 按用途分类
- 智能搜索
- 推荐配置

---

## 💡 使用场景

### 场景 1: 考驾照
```bash
python id_photo_quick.py 我的照片.jpg driving_license white
```
**输出**: 22×32mm白底驾驶证照片

### 场景 2: 考公务员
```bash
python id_photo_maker.py
# 搜索"公务员" → 选择蓝底
```
**输出**: 35×45mm蓝底公务员考试照片

### 场景 3: 办护照
```bash
python id_photo_quick.py 我的照片.jpg passport white
```
**输出**: 33×48mm白底护照照片

### 场景 4: 求职简历
```bash
python id_photo_quick.py 我的照片.jpg resume blue
```
**输出**: 25×35mm蓝底简历照片

### 场景 5: 批量制作
```python
# 运行示例代码
python example_usage.py
# 选择 "6. 批量生成多种规格"
```
**输出**: 一次生成多种规格和颜色

---

## 📖 学习路径

### 第1步: 快速上手（5分钟）
阅读 **[快速入门.md](./快速入门.md)**
- 了解3种使用方式
- 完成第一张证件照
- 掌握常用命令

### 第2步: 深入了解（15分钟）
阅读 **[ID_PHOTO_README.md](./ID_PHOTO_README.md)**
- 查看完整功能
- 学习高级技巧
- 解决常见问题

### 第3步: 技术细节（可选）
阅读 **[证件照系统优化报告.md](./证件照系统优化报告.md)**
- 了解系统架构
- 查看优化细节
- 扩展新功能

---

## ❓ 常见问题

### Q1: 首次运行很慢？
**A**: 首次会下载BiRefNet模型（约1GB），下载后会缓存到本地。

### Q2: 如何查看支持的证件？
**A**: 
```bash
# 简洁列表
python show_all_specs.py summary

# 详细信息
python show_all_specs.py
```

### Q3: 如何选择背景颜色？
**A**: 每个证件都有推荐颜色，使用交互式工具会自动提示。

### Q4: 能批量处理吗？
**A**: 可以！使用快捷命令或编程方式都支持批量处理。

### Q5: 照片质量够用吗？
**A**: 300 DPI输出，符合打印和电子版提交标准。

---

## 🎁 实用工具

### 查看规格

```bash
# 分类摘要
python show_all_specs.py summary

# 详细信息
python show_all_specs.py

# 背景颜色
python show_all_specs.py colors
```

### 搜索规格

```python
from id_photo_specs import search_specs

# 搜索"驾照"
results = search_specs("驾照")

# 搜索"考试"
results = search_specs("考试")
```

### 查看帮助

```bash
# 快捷命令帮助
python id_photo_quick.py --help

# 查看规格列表
python id_photo_quick.py --list-specs

# 查看颜色列表
python id_photo_quick.py --list-colors
```

---

## 🔥 快捷命令速查

```bash
# 驾驶证（白底）
python id_photo_quick.py photo.jpg driving_license white

# 公务员（蓝底）
python id_photo_quick.py photo.jpg civil_servant blue

# 护照（白底）
python id_photo_quick.py photo.jpg passport white

# 四六级（蓝底）
python id_photo_quick.py photo.jpg cet blue

# 一寸（多色）
python id_photo_quick.py photo.jpg 1inch white
python id_photo_quick.py photo.jpg 1inch blue
python id_photo_quick.py photo.jpg 1inch red

# 透明背景
python id_photo_quick.py photo.jpg 1inch transparent
```

---

## 📞 获取帮助

- 📖 **快速入门**: [快速入门.md](./快速入门.md)
- 📚 **完整文档**: [ID_PHOTO_README.md](./ID_PHOTO_README.md)
- 💻 **代码示例**: [example_usage.py](./example_usage.py)
- 🔍 **查看规格**: `python show_all_specs.py`
- 🐛 **问题修复**: [问题修复说明.md](./问题修复说明.md)
- ⚠️ **已知问题**: [已知问题和解决方案.md](./已知问题和解决方案.md)

---

## 🔧 故障排除

### 问题1: ModuleNotFoundError: No module named 'id_photo_specs'

**解决**: 已修复！更新到最新代码即可。

### 问题2: Server 启动时报 pymatting 错误

**解决**: 已优化为延迟加载，不影响 Server 启动。

### 问题3: Python 3.13 下生成证件照时报错

**临时方案**:
```bash
export NUMBA_CACHE_DIR=/tmp
python your_script.py
```

**推荐方案**: 使用 Python 3.11 或 3.12

**详细说明**: 查看 [已知问题和解决方案.md](./已知问题和解决方案.md)

---

## 📊 系统信息

- **版本**: v2.0
- **发布日期**: 2025-12-03
- **支持规格**: 33种
- **分类数量**: 8大类
- **背景颜色**: 4种 + 透明
- **输出质量**: 300 DPI

---

**开始制作您的第一张证件照吧！** 📸✨

```bash
python id_photo_maker.py
```
