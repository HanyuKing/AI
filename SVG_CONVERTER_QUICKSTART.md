# SVG转换器 - 快速开始指南

## ✅ 问题已解决

`pypotrace` 库因为需要编译 C 扩展而安装失败。我们已改用 `potrace` 命令行工具，更稳定可靠！

## 🚀 快速安装（3步）

### 1️⃣ 安装系统工具
```bash
# 已完成！potrace 和 agg 已安装
brew install potrace agg inkscape  # inkscape 可选，用于更好的 EPS 转换
```

### 2️⃣ 安装 Python 依赖
```bash
cd /Users/rogerswang/my/source_code/AI
pip install -r server/requirements.txt
# ✅ 已完成！所有依赖都已成功安装
```

### 3️⃣ 启动服务器
```bash
cd /Users/rogerswang/my/source_code/AI

# 推荐：使用启动脚本（已自动配置环境变量）
./run_server.sh

# 或手动设置环境变量后启动
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
python -m server.app
```

**注意**：如果遇到 Cairo 库找不到的错误，请确保设置了 `DYLD_LIBRARY_PATH` 环境变量指向 Homebrew 的库目录。

## 🌐 访问应用

打开浏览器访问：
- 主页：http://localhost:8000
- SVG转换器：http://localhost:8000/tools/svg-converter

## 🎯 功能列表

### 输入格式 → SVG
- ✅ **EPS转SVG** - 矢量格式转换
- ✅ **JPG转SVG** - 图片矢量化（3种模式）
- ✅ **PNG转SVG** - 图片矢量化（支持透明）
- ✅ **PDF转SVG** - PDF页面提取

### SVG → 输出格式
- ✅ **SVG转EPS** - 输出为印刷格式
- ✅ **SVG转JPG** - 光栅化（自动白背景）
- ✅ **SVG转PDF** - 转换为文档格式
- ✅ **SVG转PNG** - 光栅化（保持透明）

### 优化功能
- ✅ **优化SVG** - 压缩文件大小（通常减少30-60%）

## 🔧 技术细节

### 使用的工具
- **potrace 命令行** (v1.16) - 位图转矢量
- **cairosvg** - SVG 转位图格式
- **svglib + reportlab** - SVG 与 PDF 互转
- **PyMuPDF** - PDF 处理
- **scour** - SVG 优化

### 为什么不用 pypotrace？
`pypotrace` 需要编译 C 扩展，依赖 `libagg` 库，在某些环境下安装困难。使用系统的 `potrace` 命令行工具：
- ✅ 安装简单（brew install）
- ✅ 更稳定可靠
- ✅ 性能更好
- ✅ 无需处理 Python C 扩展编译问题

## 📝 API 使用示例

### 1. JPG/PNG 转 SVG
```bash
curl -X POST http://localhost:8000/api/media/svg/to-svg \
  -F "file=@image.jpg" \
  -F "trace_mode=default" \
  -o output.svg
```

### 2. SVG 转 PNG
```bash
curl -X POST http://localhost:8000/api/media/svg/from-svg \
  -F "file=@image.svg" \
  -F "target_format=PNG" \
  -F "width=800" \
  -F "height=600" \
  -o output.png
```

### 3. 优化 SVG
```bash
curl -X POST http://localhost:8000/api/media/svg/optimize \
  -F "file=@image.svg" \
  -F "precision=2" \
  -F "remove_metadata=true" \
  -o optimized.svg
```

## 🎨 矢量化模式说明

转换 JPG/PNG 为 SVG 时可选择：

- **default (默认)**: 平衡细节和文件大小
- **detailed (详细)**: 保留更多细节，适合复杂图像
- **simplified (简化)**: 减少节点数，适合简单图标

## ⚡ 性能提示

1. **图像预处理**：转 SVG 前将图片调整到合适大小（建议不超过 2000px）
2. **黑白图像**：简单的黑白图标矢量化效果最佳
3. **批量处理**：可以编写脚本调用 API 批量转换
4. **优化建议**：复杂 SVG 可以多次优化以获得更小体积

## 🐛 故障排除

### potrace 命令未找到
```bash
# 安装 potrace
brew install potrace

# 验证安装
which potrace
potrace --version
```

### cairosvg 错误
确保安装了系统的 Cairo 库：
```bash
brew install cairo
```

### 端口被占用
修改启动端口：
```python
# server/app.py
uvicorn.run("server.app:app", host="0.0.0.0", port=8001, reload=True)
```

## 📚 更多文档

详细文档请查看：`SVG_CONVERTER_README.md`

## 🎉 开始使用

现在所有依赖都已安装完成，直接启动服务器即可使用！

```bash
python -m server.app
```

访问 http://localhost:8000/tools/svg-converter 开始转换文件！

