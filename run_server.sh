#!/bin/bash
cd "$(dirname "$0")"

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置动态库路径（用于cairosvg和其他图形库）
if [ -d "/opt/homebrew/lib" ]; then
    export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
fi

echo "Starting MediaToolbox Server..."
echo "SVG Converter: http://localhost:8000/tools/svg-converter"
echo "Visit http://localhost:8000 to use all tools."
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload --reload-exclude "**/ENTER/**"

