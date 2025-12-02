"""
SVG转换服务
支持多种格式与SVG之间的互转，以及SVG优化
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from PIL import Image
import cairosvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPS
import fitz  # PyMuPDF
from xml.etree import ElementTree as ET
import base64


class SVGService:
    """SVG格式转换服务"""
    
    # ==============================================
    # 输入格式 → SVG
    # ==============================================
    
    @staticmethod
    def eps_to_svg(input_path: str, output_path: str) -> None:
        """
        EPS转SVG
        使用Inkscape命令行工具（需要系统安装Inkscape）
        """
        try:
            # 尝试使用Inkscape（最佳效果）
            subprocess.run([
                'inkscape',
                input_path,
                '--export-type=svg',
                f'--export-filename={output_path}'
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果Inkscape不可用，尝试使用cairosvg（备选方案）
            # 注意：EPS到SVG需要先转为其他格式
            try:
                # 使用PIL读取EPS，转为PNG，再转为SVG
                with Image.open(input_path) as img:
                    img.load(scale=10)  # 高分辨率加载
                    temp_png = tempfile.mktemp(suffix='.png')
                    img.save(temp_png, 'PNG', dpi=(300, 300))
                    SVGService.png_to_svg(temp_png, output_path)
                    os.unlink(temp_png)
            except Exception as e:
                raise Exception(f"EPS转SVG失败，请确保安装Inkscape或提供有效的EPS文件: {str(e)}")
    
    @staticmethod
    def jpg_to_svg(input_path: str, output_path: str, trace_mode: str = 'default') -> None:
        """
        JPG转SVG（图像矢量化）
        使用potrace进行路径追踪
        
        Args:
            input_path: 输入JPG路径
            output_path: 输出SVG路径
            trace_mode: 追踪模式 ('default', 'detailed', 'simplified')
        """
        SVGService._image_to_svg(input_path, output_path, trace_mode)
    
    @staticmethod
    def png_to_svg(input_path: str, output_path: str, trace_mode: str = 'default') -> None:
        """
        PNG转SVG（图像矢量化）
        使用potrace进行路径追踪
        
        Args:
            input_path: 输入PNG路径
            output_path: 输出SVG路径
            trace_mode: 追踪模式 ('default', 'detailed', 'simplified')
        """
        SVGService._image_to_svg(input_path, output_path, trace_mode)
    
    @staticmethod
    def _image_to_svg(input_path: str, output_path: str, trace_mode: str = 'default') -> None:
        """
        通用图像转SVG方法（使用potrace命令行工具）
        """
        try:
            # 使用potrace命令行工具（更稳定的方案）
            # 首先将图像转为PBM格式（potrace需要）
            temp_pbm = tempfile.mktemp(suffix='.pbm')
            
            with Image.open(input_path) as img:
                # 转为灰度图
                img_gray = img.convert('L')
                
                # 根据模式调整阈值
                threshold = 128
                if trace_mode == 'detailed':
                    threshold = 100
                elif trace_mode == 'simplified':
                    threshold = 150
                
                # 转为黑白位图
                img_bw = img_gray.point(lambda x: 255 if x > threshold else 0, mode='1')
                
                # 保存为PBM格式
                img_bw.save(temp_pbm)
            
            # 构建potrace命令
            potrace_args = ['potrace', '-s', '-o', output_path]
            
            # 根据模式添加参数
            if trace_mode == 'detailed':
                potrace_args.extend(['-t', '2', '-O', '0.2'])  # 更低的阈值，更多细节
            elif trace_mode == 'simplified':
                potrace_args.extend(['-t', '5', '-O', '1.0'])  # 更高的阈值，更简化
            else:
                potrace_args.extend(['-t', '3'])  # 默认阈值
            
            potrace_args.append(temp_pbm)
            
            # 执行potrace
            result = subprocess.run(
                potrace_args,
                capture_output=True,
                text=True,
                check=True
            )
            
            # 清理临时文件
            if os.path.exists(temp_pbm):
                os.unlink(temp_pbm)
                    
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # 清理临时文件
            if 'temp_pbm' in locals() and os.path.exists(temp_pbm):
                os.unlink(temp_pbm)
            
            # 如果potrace命令行不可用，使用备选方案（嵌入图像）
            SVGService._embed_image_in_svg(input_path, output_path)
        except Exception as e:
            # 清理临时文件
            if 'temp_pbm' in locals() and os.path.exists(temp_pbm):
                os.unlink(temp_pbm)
            raise Exception(f"图像转SVG失败: {str(e)}")
    
    @staticmethod
    def _embed_image_in_svg(input_path: str, output_path: str) -> None:
        """将图像嵌入到SVG中（备选方案）"""
        with Image.open(input_path) as img:
            width, height = img.size
            
            # 将图像转为base64
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # 创建SVG
            svg_content = f'''<?xml version="1.0" standalone="no"?>
<svg width="{width}" height="{height}" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <image width="{width}" height="{height}" xlink:href="data:image/png;base64,{img_base64}"/>
</svg>'''
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
    
    @staticmethod
    def pdf_to_svg(input_path: str, output_path: str, page_num: int = 0, 
                   merge_pages: bool = False) -> None:
        """
        PDF转SVG
        可以转换单页或将多页合并
        
        Args:
            input_path: 输入PDF路径
            output_path: 输出SVG路径
            page_num: 要转换的页码（从0开始）
            merge_pages: 是否合并所有页面
        """
        try:
            doc = fitz.open(input_path)
            
            if merge_pages:
                # 合并所有页面到一个SVG
                svg_parts = []
                total_height = 0
                max_width = 0
                
                # 收集所有页面的SVG
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    rect = page.rect
                    svg_data = page.get_svg_image()
                    
                    # 解析SVG以提取内容
                    root = ET.fromstring(svg_data)
                    
                    # 调整Y坐标
                    group = ET.Element('g', transform=f'translate(0, {total_height})')
                    for child in root:
                        if child.tag.endswith('g'):
                            group.append(child)
                    
                    svg_parts.append(group)
                    total_height += rect.height
                    max_width = max(max_width, rect.width)
                
                # 创建合并的SVG
                merged_svg = ET.Element('svg', {
                    'width': str(max_width),
                    'height': str(total_height),
                    'xmlns': 'http://www.w3.org/2000/svg'
                })
                
                for part in svg_parts:
                    merged_svg.append(part)
                
                # 写入文件
                tree = ET.ElementTree(merged_svg)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
            else:
                # 转换单页
                page = doc[page_num]
                svg_data = page.get_svg_image()
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(svg_data)
            
            doc.close()
            
        except Exception as e:
            raise Exception(f"PDF转SVG失败: {str(e)}")
    
    # ==============================================
    # SVG → 输出格式
    # ==============================================
    
    @staticmethod
    def svg_to_png(input_path: str, output_path: str, width: Optional[int] = None,
                   height: Optional[int] = None, dpi: int = 96) -> None:
        """
        SVG转PNG
        
        Args:
            input_path: 输入SVG路径
            output_path: 输出PNG路径
            width: 输出宽度（可选）
            height: 输出高度（可选）
            dpi: 分辨率
        """
        try:
            kwargs = {'dpi': dpi}
            if width:
                kwargs['output_width'] = width
            if height:
                kwargs['output_height'] = height
            
            cairosvg.svg2png(url=input_path, write_to=output_path, **kwargs)
        except Exception as e:
            raise Exception(f"SVG转PNG失败: {str(e)}")
    
    @staticmethod
    def svg_to_jpg(input_path: str, output_path: str, width: Optional[int] = None,
                   height: Optional[int] = None, quality: int = 95) -> None:
        """
        SVG转JPG
        先转PNG再转JPG
        
        Args:
            input_path: 输入SVG路径
            output_path: 输出JPG路径
            width: 输出宽度（可选）
            height: 输出高度（可选）
            quality: JPG质量(1-100)
        """
        try:
            # 先转为PNG
            temp_png = tempfile.mktemp(suffix='.png')
            SVGService.svg_to_png(input_path, temp_png, width, height)
            
            # PNG转JPG
            with Image.open(temp_png) as img:
                # 如果有透明通道，添加白色背景
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                else:
                    img = img.convert('RGB')
                
                img.save(output_path, 'JPEG', quality=quality)
            
            # 清理临时文件
            os.unlink(temp_png)
            
        except Exception as e:
            raise Exception(f"SVG转JPG失败: {str(e)}")
    
    @staticmethod
    def svg_to_pdf(input_path: str, output_path: str, merge_mode: bool = False) -> None:
        """
        SVG转PDF
        
        Args:
            input_path: 输入SVG路径
            output_path: 输出PDF路径
            merge_mode: 是否作为单独的页面（暂不支持）
        """
        try:
            # 使用svglib和reportlab
            drawing = svg2rlg(input_path)
            if drawing is None:
                raise Exception("无法解析SVG文件")
            
            renderPDF.drawToFile(drawing, output_path)
            
        except Exception as e:
            # 备选方案：使用cairosvg
            try:
                cairosvg.svg2pdf(url=input_path, write_to=output_path)
            except Exception as e2:
                raise Exception(f"SVG转PDF失败: {str(e)}, 备选方案也失败: {str(e2)}")
    
    @staticmethod
    def svg_to_eps(input_path: str, output_path: str) -> None:
        """
        SVG转EPS
        使用cairosvg或svglib
        """
        try:
            # 尝试使用cairosvg
            cairosvg.svg2ps(url=input_path, write_to=output_path)
        except Exception as e:
            # 备选方案：使用svglib
            try:
                drawing = svg2rlg(input_path)
                if drawing is None:
                    raise Exception("无法解析SVG文件")
                
                renderPS.drawToFile(drawing, output_path)
            except Exception as e2:
                raise Exception(f"SVG转EPS失败: {str(e)}, 备选方案也失败: {str(e2)}")
    
    # ==============================================
    # SVG优化
    # ==============================================
    
    @staticmethod
    def optimize_svg(input_path: str, output_path: str, 
                     precision: int = 2, 
                     remove_metadata: bool = True,
                     remove_comments: bool = True,
                     indent_type: str = 'space') -> None:
        """
        优化SVG文件
        使用scour或自定义优化
        
        Args:
            input_path: 输入SVG路径
            output_path: 输出SVG路径
            precision: 数值精度
            remove_metadata: 是否移除元数据
            remove_comments: 是否移除注释
            indent_type: 缩进类型 ('space', 'tab', 'none')
        """
        try:
            # 尝试使用scour命令行工具
            scour_options = [
                'scour',
                '-i', input_path,
                '-o', output_path,
                f'--set-precision={precision}',
                '--enable-viewboxing',
                '--enable-id-stripping',
                '--enable-comment-stripping' if remove_comments else '',
                '--remove-metadata' if remove_metadata else '',
                '--strip-xml-space',
            ]
            
            # 过滤空选项
            scour_options = [opt for opt in scour_options if opt]
            
            if indent_type == 'none':
                scour_options.append('--indent=none')
            elif indent_type == 'tab':
                scour_options.append('--indent=tab')
            else:
                scour_options.append('--indent=space')
            
            subprocess.run(scour_options, check=True, capture_output=True)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 如果scour不可用，使用简单的XML优化
            SVGService._simple_svg_optimize(input_path, output_path, precision, 
                                           remove_metadata, remove_comments)
    
    @staticmethod
    def _simple_svg_optimize(input_path: str, output_path: str, 
                            precision: int = 2,
                            remove_metadata: bool = True,
                            remove_comments: bool = True) -> None:
        """简单的SVG优化（不依赖scour）"""
        try:
            # 解析SVG
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            # 移除注释
            if remove_comments:
                for element in root.iter():
                    # 移除注释节点（需要特殊处理）
                    pass
            
            # 移除元数据
            if remove_metadata:
                namespaces = {
                    'svg': 'http://www.w3.org/2000/svg',
                    'dc': 'http://purl.org/dc/elements/1.1/',
                    'cc': 'http://creativecommons.org/ns#',
                    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                }
                
                for ns_prefix, ns_uri in namespaces.items():
                    for metadata in root.findall(f'.//{{{ns_uri}}}metadata'):
                        root.remove(metadata)
                    for desc in root.findall(f'.//{{{ns_uri}}}desc'):
                        root.remove(desc)
            
            # 优化数值精度
            SVGService._round_svg_numbers(root, precision)
            
            # 写入文件
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            # 如果优化失败，直接复制文件
            import shutil
            shutil.copy2(input_path, output_path)
    
    @staticmethod
    def _round_svg_numbers(element, precision: int = 2):
        """递归优化SVG中的数值精度"""
        import re
        
        # 需要优化的属性
        numeric_attrs = ['x', 'y', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry',
                        'x1', 'y1', 'x2', 'y2', 'stroke-width', 'font-size']
        
        for attr in numeric_attrs:
            if attr in element.attrib:
                try:
                    value = float(element.attrib[attr])
                    element.attrib[attr] = f'{value:.{precision}f}'
                except ValueError:
                    pass
        
        # 优化path的d属性
        if 'd' in element.attrib:
            path_data = element.attrib['d']
            # 简单的数值替换
            def round_match(match):
                try:
                    num = float(match.group(0))
                    return f'{num:.{precision}f}'
                except:
                    return match.group(0)
            
            path_data = re.sub(r'-?\d+\.\d+', round_match, path_data)
            element.attrib['d'] = path_data
        
        # 递归处理子元素
        for child in element:
            SVGService._round_svg_numbers(child, precision)
    
    # ==============================================
    # 辅助方法
    # ==============================================
    
    @staticmethod
    def get_svg_info(input_path: str) -> dict:
        """
        获取SVG文件信息
        
        Returns:
            包含宽度、高度、文件大小等信息的字典
        """
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            # 获取尺寸
            width = root.get('width', 'auto')
            height = root.get('height', 'auto')
            viewBox = root.get('viewBox', '')
            
            # 获取文件大小
            file_size = os.path.getsize(input_path)
            
            return {
                'width': width,
                'height': height,
                'viewBox': viewBox,
                'file_size': file_size,
                'file_size_kb': round(file_size / 1024, 2)
            }
        except Exception as e:
            raise Exception(f"获取SVG信息失败: {str(e)}")

