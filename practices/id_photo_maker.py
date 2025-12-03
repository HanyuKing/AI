"""
智能证件照制作 - 交互式命令行工具
提供友好的用户体验，支持场景分类、搜索和快速选择
"""

import os
import sys
from typing import Optional, Tuple
from PIL import Image

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .id_photo_utils import IdPhotoProcessor
    from .id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS, get_spec_by_id, search_specs
except ImportError:
    from id_photo_utils import IdPhotoProcessor
    from id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS, get_spec_by_id, search_specs


class IdPhotoMaker:
    """智能证件照制作器"""
    
    def __init__(self):
        self.processor: Optional[IdPhotoProcessor] = None
        self.input_image_path: Optional[str] = None
        
    def print_header(self):
        """打印欢迎标题"""
        print("\n" + "="*60)
        print("📸 智能证件照制作系统")
        print("="*60)
        
    def print_categories(self):
        """显示所有场景分类"""
        print("\n【证件照场景分类】\n")
        
        for idx, (cat_id, cat_data) in enumerate(ID_PHOTO_SPECS.items(), 1):
            icon = cat_data["icon"]
            name = cat_data["category_name"]
            count = len(cat_data["specs"])
            print(f"{idx}. {icon} {name} ({count}种规格)")
        
        print(f"\n0. 🔍 搜索规格")
        print(f"q. 退出系统")
        
    def show_category_specs(self, category_id: str):
        """显示某个分类下的所有规格"""
        cat_data = ID_PHOTO_SPECS[category_id]
        print(f"\n【{cat_data['icon']} {cat_data['category_name']}】\n")
        
        specs = []
        for spec_id, spec_info in cat_data["specs"].items():
            specs.append((spec_id, spec_info))
        
        for idx, (spec_id, spec_info) in enumerate(specs, 1):
            name = spec_info["name"]
            size_mm = spec_info["size_mm"]
            usage = spec_info["usage"]
            bg_colors = ", ".join([BG_COLOR_PRESETS[c]["name"] for c in spec_info["common_bg"]])
            
            print(f"{idx}. {name}")
            print(f"   尺寸: {size_mm[0]}×{size_mm[1]} mm")
            print(f"   用途: {usage}")
            print(f"   常用底色: {bg_colors}")
            
            if "note" in spec_info:
                print(f"   ⚠️  注意: {spec_info['note']}")
            print()
        
        return specs
    
    def search_specs_interactive(self):
        """交互式搜索规格"""
        keyword = input("\n🔍 请输入搜索关键词（如：驾照、公务员、护照）: ").strip()
        
        if not keyword:
            print("❌ 搜索关键词不能为空")
            return None
        
        results = search_specs(keyword)
        
        if not results:
            print(f"😕 未找到包含「{keyword}」的规格")
            return None
        
        print(f"\n【搜索结果：找到 {len(results)} 个匹配项】\n")
        
        for idx, result in enumerate(results, 1):
            print(f"{idx}. {result['category_icon']} {result['name']} ({result['category']})")
            print(f"   尺寸: {result['size_mm'][0]}×{result['size_mm'][1]} mm")
            print(f"   用途: {result['usage']}")
            print()
        
        return results
    
    def select_spec(self) -> Optional[Tuple[str, dict]]:
        """选择证件照规格，返回 (spec_id, spec_info)"""
        while True:
            self.print_categories()
            choice = input("\n请选择场景分类（输入编号）: ").strip().lower()
            
            if choice == 'q':
                return None
            
            if choice == '0':
                # 搜索模式
                results = self.search_specs_interactive()
                if not results:
                    continue
                
                spec_choice = input("\n请选择规格（输入编号，0返回上级）: ").strip()
                if spec_choice == '0':
                    continue
                
                try:
                    idx = int(spec_choice) - 1
                    if 0 <= idx < len(results):
                        selected = results[idx]
                        return (selected["id"], selected)
                    else:
                        print("❌ 无效的选择")
                except ValueError:
                    print("❌ 请输入有效的编号")
                continue
            
            # 选择分类
            try:
                cat_idx = int(choice) - 1
                categories = list(ID_PHOTO_SPECS.keys())
                if 0 <= cat_idx < len(categories):
                    cat_id = categories[cat_idx]
                    specs = self.show_category_specs(cat_id)
                    
                    spec_choice = input("请选择规格（输入编号，0返回上级）: ").strip()
                    if spec_choice == '0':
                        continue
                    
                    try:
                        spec_idx = int(spec_choice) - 1
                        if 0 <= spec_idx < len(specs):
                            spec_id, spec_info = specs[spec_idx]
                            return (spec_id, spec_info)
                        else:
                            print("❌ 无效的选择")
                    except ValueError:
                        print("❌ 请输入有效的编号")
                else:
                    print("❌ 无效的分类编号")
            except ValueError:
                print("❌ 请输入有效的编号")
    
    def select_background_color(self, spec_info: dict) -> str:
        """选择背景颜色"""
        print("\n【选择背景颜色】\n")
        
        # 显示推荐的背景颜色
        common_bg = spec_info["common_bg"]
        print("💡 推荐颜色:")
        for idx, color_name in enumerate(common_bg, 1):
            color_info = BG_COLOR_PRESETS[color_name]
            print(f"  {idx}. {color_info['name']} - {color_info['description']}")
        
        print("\n📌 其他颜色:")
        other_colors = [c for c in BG_COLOR_PRESETS.keys() if c not in common_bg]
        start_idx = len(common_bg) + 1
        for idx, color_name in enumerate(other_colors, start_idx):
            color_info = BG_COLOR_PRESETS[color_name]
            print(f"  {idx}. {color_info['name']} - {color_info['description']}")
        
        print(f"  {start_idx + len(other_colors)}. 透明背景（PNG格式）")
        
        while True:
            choice = input(f"\n请选择背景颜色（1-{start_idx + len(other_colors)}）: ").strip()
            try:
                choice_idx = int(choice)
                all_colors = common_bg + other_colors
                
                if 1 <= choice_idx <= len(all_colors):
                    return all_colors[choice_idx - 1]
                elif choice_idx == start_idx + len(other_colors):
                    return "transparent"
                else:
                    print("❌ 无效的选择")
            except ValueError:
                print("❌ 请输入有效的编号")
    
    def confirm_settings(self, spec_info: dict, bg_color_name: str, use_beautify: bool) -> bool:
        """确认设置"""
        print("\n" + "="*60)
        print("【确认制作设置】")
        print("="*60)
        print(f"📄 证件类型: {spec_info['name']}")
        print(f"📏 照片尺寸: {spec_info['size_mm'][0]}×{spec_info['size_mm'][1]} mm")
        print(f"🎨 背景颜色: {BG_COLOR_PRESETS.get(bg_color_name, {}).get('name', '透明背景')}")
        print(f"✨ 美颜处理: {'开启' if use_beautify else '关闭'}")
        print(f"💾 输出DPI: 300")
        print(f"📂 输入文件: {self.input_image_path}")
        print("="*60)
        
        confirm = input("\n确认制作？(y/n): ").strip().lower()
        return confirm == 'y'
    
    def make_id_photo(self):
        """主流程：制作证件照"""
        self.print_header()
        
        # 1. 选择输入图片
        print("\n【步骤 1/4】选择照片")
        self.input_image_path = input("请输入照片路径: ").strip()
        
        if not os.path.exists(self.input_image_path):
            print(f"❌ 文件不存在: {self.input_image_path}")
            return
        
        try:
            self.processor = IdPhotoProcessor(self.input_image_path)
            print("✅ 照片加载成功")
        except Exception as e:
            print(f"❌ 加载照片失败: {e}")
            return
        
        # 2. 选择证件照规格
        print("\n【步骤 2/4】选择证件类型")
        result = self.select_spec()
        if result is None:
            print("👋 已退出")
            return
        
        spec_id, spec_info = result
        print(f"✅ 已选择: {spec_info['name']}")
        
        # 3. 选择背景颜色
        print("\n【步骤 3/4】选择背景颜色")
        bg_color_name = self.select_background_color(spec_info)
        print(f"✅ 已选择: {BG_COLOR_PRESETS.get(bg_color_name, {}).get('name', '透明背景')}")
        
        # 4. 美颜选项
        print("\n【步骤 4/4】美颜设置")
        beautify = input("是否开启美颜？(y/n，默认y): ").strip().lower()
        use_beautify = beautify != 'n'
        
        # 5. 确认设置
        if not self.confirm_settings(spec_info, bg_color_name, use_beautify):
            print("❌ 已取消制作")
            return
        
        # 6. 开始制作
        print("\n⏳ 正在制作证件照...")
        print("   - 人像识别与抠图...")
        
        try:
            # 生成证件照
            if bg_color_name == "transparent":
                result_img = self.processor.generate_id_photo(
                    spec_id=spec_id,
                    bg_color=None,
                    use_beautify=use_beautify
                )
                file_ext = "png"
            else:
                result_img = self.processor.generate_id_photo(
                    spec_id=spec_id,
                    bg_color_name=bg_color_name,
                    use_beautify=use_beautify
                )
                file_ext = "jpg"
            
            print("   - 调整尺寸与美化...")
            
            # 保存结果
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
            output_filename = f"{base_name}_{spec_info['name']}_{bg_color_name if bg_color_name != 'transparent' else '透明'}.{file_ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            if file_ext == "jpg":
                result_img.save(output_path, "JPEG", quality=95, dpi=(300, 300))
            else:
                result_img.save(output_path, "PNG", dpi=(300, 300))
            
            print("\n" + "="*60)
            print("✅ 证件照制作完成！")
            print("="*60)
            print(f"📂 保存位置: {output_path}")
            print(f"📏 照片尺寸: {spec_info['size_mm'][0]}×{spec_info['size_mm'][1]} mm")
            print(f"📐 像素尺寸: {result_img.width}×{result_img.height} px")
            print(f"💾 文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ 制作失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """运行主程序"""
        try:
            self.make_id_photo()
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，已退出")
        except Exception as e:
            print(f"\n❌ 程序错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """命令行入口"""
    maker = IdPhotoMaker()
    maker.run()


if __name__ == "__main__":
    main()
