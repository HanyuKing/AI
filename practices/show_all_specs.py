"""
显示所有证件照规格
快速查看系统支持的所有证件照类型
"""

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS
except ImportError:
    from id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS


def show_all_specs():
    """显示所有证件照规格"""
    print("\n" + "="*80)
    print("📸 证件照规格大全 - 共计 {} 个分类".format(len(ID_PHOTO_SPECS)))
    print("="*80)
    
    total_specs = 0
    
    for cat_id, cat_data in ID_PHOTO_SPECS.items():
        icon = cat_data["icon"]
        name = cat_data["category_name"]
        specs = cat_data["specs"]
        
        print(f"\n{icon} {name} ({len(specs)} 种规格)")
        print("-" * 80)
        
        for spec_id, spec_info in specs.items():
            total_specs += 1
            print(f"\n  📄 {spec_info['name']}")
            print(f"     ID: {spec_id}")
            print(f"     尺寸: {spec_info['size_mm'][0]}×{spec_info['size_mm'][1]} mm " +
                  f"({spec_info['size_px'][0]}×{spec_info['size_px'][1]} px @ 300 DPI)")
            print(f"     用途: {spec_info['usage']}")
            print(f"     说明: {spec_info['description']}")
            
            # 显示常用背景色
            bg_colors = []
            for color_name in spec_info['common_bg']:
                color_info = BG_COLOR_PRESETS[color_name]
                bg_colors.append(f"{color_info['name']}({color_info['hex']})")
            print(f"     常用底色: {', '.join(bg_colors)}")
            
            # 显示注意事项
            if "note" in spec_info:
                print(f"     ⚠️  注意: {spec_info['note']}")
    
    print("\n" + "="*80)
    print(f"✅ 总计支持 {total_specs} 种证件照规格")
    print("="*80)


def show_categories_summary():
    """显示分类摘要"""
    print("\n" + "="*80)
    print("📋 证件照分类摘要")
    print("="*80 + "\n")
    
    for idx, (cat_id, cat_data) in enumerate(ID_PHOTO_SPECS.items(), 1):
        icon = cat_data["icon"]
        name = cat_data["category_name"]
        count = len(cat_data["specs"])
        
        # 列出该分类下的所有规格名称
        spec_names = [spec_info["name"] for spec_info in cat_data["specs"].values()]
        
        print(f"{idx}. {icon} {name} ({count} 种)")
        print(f"   包含: {', '.join(spec_names)}")
        print()
    
    print("="*80)


def show_background_colors():
    """显示所有背景颜色"""
    print("\n" + "="*80)
    print("🎨 可用背景颜色")
    print("="*80 + "\n")
    
    for color_name, color_info in BG_COLOR_PRESETS.items():
        print(f"🎨 {color_info['name']}")
        print(f"   ID: {color_name}")
        print(f"   RGB: {color_info['rgb']}")
        print(f"   HEX: {color_info['hex']}")
        print(f"   说明: {color_info['description']}")
        print()
    
    print("="*80)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "summary":
            show_categories_summary()
        elif command == "colors":
            show_background_colors()
        elif command == "all":
            show_all_specs()
        else:
            print(f"未知命令: {command}")
            print("\n使用方法:")
            print("  python show_all_specs.py          # 显示所有详细规格")
            print("  python show_all_specs.py summary  # 显示分类摘要")
            print("  python show_all_specs.py colors   # 显示背景颜色")
    else:
        # 默认显示详细规格
        show_all_specs()


if __name__ == "__main__":
    main()
