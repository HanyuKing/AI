"""
证件照制作 - 使用示例
演示如何通过编程方式快速生成各种证件照
"""

import os

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .id_photo_utils import IdPhotoProcessor
    from .id_photo_specs import search_specs, get_spec_by_id
except ImportError:
    from id_photo_utils import IdPhotoProcessor
    from id_photo_specs import search_specs, get_spec_by_id


def example_1_driving_license():
    """示例 1: 制作驾驶证照片（白底）"""
    print("\n" + "="*60)
    print("示例 1: 制作驾驶证照片")
    print("="*60)
    
    # 输入照片路径
    input_path = "images/工卡照.jpeg"
    
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return
    
    # 创建处理器
    processor = IdPhotoProcessor(input_path)
    
    # 生成驾驶证照片（白底，开启美颜）
    result = processor.generate_id_photo(
        spec_id="driving_license",
        bg_color_name="white",
        use_beautify=True
    )
    
    # 保存结果
    output_path = "output/example_驾驶证_白底.jpg"
    os.makedirs("output", exist_ok=True)
    result.save(output_path, "JPEG", quality=95, dpi=(300, 300))
    
    print(f"✅ 已生成: {output_path}")
    print(f"   尺寸: {result.width}×{result.height} px (22×32 mm @ 300 DPI)")


def example_2_civil_servant():
    """示例 2: 制作公务员考试照片（蓝底）"""
    print("\n" + "="*60)
    print("示例 2: 制作公务员考试照片")
    print("="*60)
    
    input_path = "images/工卡照.jpeg"
    
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return
    
    processor = IdPhotoProcessor(input_path)
    
    # 生成公务员考试照片（蓝底，开启美颜）
    result = processor.generate_id_photo(
        spec_id="civil_servant",
        bg_color_name="blue",
        use_beautify=True
    )
    
    output_path = "output/example_公务员考试_蓝底.jpg"
    os.makedirs("output", exist_ok=True)
    result.save(output_path, "JPEG", quality=95, dpi=(300, 300))
    
    print(f"✅ 已生成: {output_path}")
    print(f"   尺寸: {result.width}×{result.height} px (35×45 mm @ 300 DPI)")


def example_3_passport():
    """示例 3: 制作护照照片（白底）"""
    print("\n" + "="*60)
    print("示例 3: 制作护照照片")
    print("="*60)
    
    input_path = "images/工卡照.jpeg"
    
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return
    
    processor = IdPhotoProcessor(input_path)
    
    # 生成护照照片（白底，开启美颜）
    result = processor.generate_id_photo(
        spec_id="passport",
        bg_color_name="white",
        use_beautify=True
    )
    
    output_path = "output/example_护照_白底.jpg"
    os.makedirs("output", exist_ok=True)
    result.save(output_path, "JPEG", quality=95, dpi=(300, 300))
    
    print(f"✅ 已生成: {output_path}")
    print(f"   尺寸: {result.width}×{result.height} px (33×48 mm @ 300 DPI)")


def example_4_transparent_bg():
    """示例 4: 制作透明背景证件照"""
    print("\n" + "="*60)
    print("示例 4: 制作透明背景证件照")
    print("="*60)
    
    input_path = "images/工卡照.jpeg"
    
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return
    
    processor = IdPhotoProcessor(input_path)
    
    # 生成一寸照片（透明背景，开启美颜）
    result = processor.generate_id_photo(
        spec_id="1inch",
        bg_color=None,  # 透明背景
        use_beautify=True
    )
    
    output_path = "output/example_一寸_透明.png"
    os.makedirs("output", exist_ok=True)
    result.save(output_path, "PNG", dpi=(300, 300))
    
    print(f"✅ 已生成: {output_path}")
    print(f"   尺寸: {result.width}×{result.height} px (25×35 mm @ 300 DPI)")


def example_5_search_and_create():
    """示例 5: 通过搜索找到规格并制作"""
    print("\n" + "="*60)
    print("示例 5: 搜索并制作证件照")
    print("="*60)
    
    # 搜索包含"考试"的规格
    results = search_specs("考试")
    print(f"\n搜索「考试」找到 {len(results)} 个规格:")
    for r in results:
        print(f"  - {r['name']}: {r['size_mm']} mm")
    
    # 选择第一个结果（公务员考试）
    if results:
        spec_id = results[0]["id"]
        print(f"\n选择制作: {results[0]['name']}")
        
        input_path = "images/工卡照.jpeg"
        
        if not os.path.exists(input_path):
            print(f"❌ 文件不存在: {input_path}")
            return
        
        processor = IdPhotoProcessor(input_path)
        
        # 生成证件照
        result = processor.generate_id_photo(
            spec_id=spec_id,
            bg_color_name="blue",
            use_beautify=True
        )
        
        output_path = f"output/example_搜索_{results[0]['name']}.jpg"
        os.makedirs("output", exist_ok=True)
        result.save(output_path, "JPEG", quality=95, dpi=(300, 300))
        
        print(f"✅ 已生成: {output_path}")


def example_6_batch_processing():
    """示例 6: 批量生成多种规格"""
    print("\n" + "="*60)
    print("示例 6: 批量生成多种规格")
    print("="*60)
    
    input_path = "images/工卡照.jpeg"
    
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return
    
    # 要生成的规格列表
    specs_to_generate = [
        ("1inch", "white", "一寸白底"),
        ("1inch", "blue", "一寸蓝底"),
        ("1inch", "red", "一寸红底"),
        ("2inch", "blue", "二寸蓝底"),
        ("passport", "white", "护照白底"),
    ]
    
    print(f"\n准备批量生成 {len(specs_to_generate)} 张证件照...")
    
    for spec_id, bg_color, desc in specs_to_generate:
        try:
            processor = IdPhotoProcessor(input_path)
            
            result = processor.generate_id_photo(
                spec_id=spec_id,
                bg_color_name=bg_color,
                use_beautify=True
            )
            
            output_path = f"output/batch_{desc}.jpg"
            os.makedirs("output", exist_ok=True)
            result.save(output_path, "JPEG", quality=95, dpi=(300, 300))
            
            print(f"✅ {desc}: {output_path}")
            
        except Exception as e:
            print(f"❌ {desc}: {e}")
    
    print(f"\n批量生成完成！")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("📸 证件照制作 - 使用示例")
    print("="*60)
    print("\n请选择要运行的示例:")
    print("  1. 制作驾驶证照片（白底）")
    print("  2. 制作公务员考试照片（蓝底）")
    print("  3. 制作护照照片（白底）")
    print("  4. 制作透明背景证件照")
    print("  5. 通过搜索找到规格并制作")
    print("  6. 批量生成多种规格")
    print("  7. 运行所有示例")
    print("  0. 退出")
    
    choice = input("\n请输入选项（0-7）: ").strip()
    
    if choice == "1":
        example_1_driving_license()
    elif choice == "2":
        example_2_civil_servant()
    elif choice == "3":
        example_3_passport()
    elif choice == "4":
        example_4_transparent_bg()
    elif choice == "5":
        example_5_search_and_create()
    elif choice == "6":
        example_6_batch_processing()
    elif choice == "7":
        example_1_driving_license()
        example_2_civil_servant()
        example_3_passport()
        example_4_transparent_bg()
        example_5_search_and_create()
        example_6_batch_processing()
    elif choice == "0":
        print("\n👋 再见！")
    else:
        print("\n❌ 无效的选项")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
