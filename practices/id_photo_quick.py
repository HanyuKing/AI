#!/usr/bin/env python3
"""
证件照快速制作 - 命令行快捷工具
用法: python id_photo_quick.py <输入图片> <规格ID> <背景颜色> [--no-beautify]

示例:
  python id_photo_quick.py photo.jpg driving_license white
  python id_photo_quick.py photo.jpg civil_servant blue
  python id_photo_quick.py photo.jpg passport white --no-beautify
"""

import sys
import os
import argparse

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .id_photo_specs import get_spec_by_id, BG_COLOR_PRESETS, ID_PHOTO_SPECS
except ImportError:
    from id_photo_specs import get_spec_by_id, BG_COLOR_PRESETS, ID_PHOTO_SPECS


def list_all_specs():
    """列出所有可用的规格ID"""
    print("\n可用的证件照规格ID:\n")
    for cat_id, cat_data in ID_PHOTO_SPECS.items():
        print(f"{cat_data['icon']} {cat_data['category_name']}:")
        for spec_id, spec_info in cat_data["specs"].items():
            print(f"  {spec_id:25} - {spec_info['name']} ({spec_info['size_mm'][0]}×{spec_info['size_mm'][1]} mm)")
        print()


def list_colors():
    """列出所有可用的背景颜色"""
    print("\n可用的背景颜色:\n")
    for color_name, color_info in BG_COLOR_PRESETS.items():
        print(f"  {color_name:15} - {color_info['name']} {color_info['hex']}")
    print(f"  {'transparent':15} - 透明背景")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="证件照快速制作工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s photo.jpg driving_license white
  %(prog)s photo.jpg civil_servant blue
  %(prog)s photo.jpg passport white --no-beautify
  
查看所有规格:
  %(prog)s --list-specs
  
查看所有颜色:
  %(prog)s --list-colors
"""
    )
    
    parser.add_argument("input_image", nargs="?", help="输入图片路径")
    parser.add_argument("spec_id", nargs="?", help="证件照规格ID（如 driving_license, passport）")
    parser.add_argument("bg_color", nargs="?", help="背景颜色（如 white, blue, red, transparent）")
    parser.add_argument("--no-beautify", action="store_true", help="不使用美颜功能")
    parser.add_argument("--output", "-o", help="输出文件路径（可选）")
    parser.add_argument("--dpi", type=int, default=300, help="输出DPI（默认300）")
    parser.add_argument("--list-specs", action="store_true", help="列出所有可用的规格ID")
    parser.add_argument("--list-colors", action="store_true", help="列出所有可用的背景颜色")
    
    args = parser.parse_args()
    
    # 显示帮助信息
    if args.list_specs:
        list_all_specs()
        return 0
    
    if args.list_colors:
        list_colors()
        return 0
    
    # 检查必需参数
    if not args.input_image or not args.spec_id or not args.bg_color:
        parser.print_help()
        return 1
    
    # 验证输入文件
    if not os.path.exists(args.input_image):
        print(f"❌ 错误: 文件不存在 - {args.input_image}")
        return 1
    
    # 验证规格ID
    spec = get_spec_by_id(args.spec_id)
    if spec is None:
        print(f"❌ 错误: 无效的规格ID - {args.spec_id}")
        print("\n使用 --list-specs 查看所有可用的规格ID")
        return 1
    
    # 验证背景颜色
    if args.bg_color == "transparent":
        bg_color = None
        bg_color_name = None
        file_ext = "png"
    elif args.bg_color in BG_COLOR_PRESETS:
        bg_color = None
        bg_color_name = args.bg_color
        file_ext = "jpg"
    else:
        print(f"❌ 错误: 无效的背景颜色 - {args.bg_color}")
        print("\n使用 --list-colors 查看所有可用的背景颜色")
        return 1
    
    # 生成输出文件名
    if args.output:
        output_path = args.output
    else:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_filename = f"{base_name}_{spec['name']}_{args.bg_color}.{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
    
    # 打印处理信息
    print("\n" + "="*60)
    print("📸 证件照快速制作")
    print("="*60)
    print(f"📄 输入图片: {args.input_image}")
    print(f"📋 证件类型: {spec['name']}")
    print(f"📏 照片尺寸: {spec['size_mm'][0]}×{spec['size_mm'][1]} mm")
    print(f"🎨 背景颜色: {BG_COLOR_PRESETS.get(args.bg_color, {}).get('name', '透明背景')}")
    print(f"✨ 美颜处理: {'关闭' if args.no_beautify else '开启'}")
    print(f"💾 输出DPI: {args.dpi}")
    print("="*60)
    
    # 开始处理
    print("\n⏳ 正在处理...")
    
    try:
        # 延迟导入，避免在查看列表时加载重量级库
        try:
            from .id_photo_utils import IdPhotoProcessor
        except ImportError:
            from id_photo_utils import IdPhotoProcessor
        
        # 创建处理器
        processor = IdPhotoProcessor(args.input_image)
        
        # 生成证件照
        result = processor.generate_id_photo(
            spec_id=args.spec_id,
            bg_color=bg_color,
            bg_color_name=bg_color_name,
            use_beautify=not args.no_beautify,
            dpi=args.dpi
        )
        
        # 保存结果
        if file_ext == "jpg":
            result.save(output_path, "JPEG", quality=95, dpi=(args.dpi, args.dpi))
        else:
            result.save(output_path, "PNG", dpi=(args.dpi, args.dpi))
        
        # 打印结果
        print("\n" + "="*60)
        print("✅ 制作完成！")
        print("="*60)
        print(f"📂 保存位置: {output_path}")
        print(f"📐 像素尺寸: {result.width}×{result.height} px")
        print(f"💾 文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
