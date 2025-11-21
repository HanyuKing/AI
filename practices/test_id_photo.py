import unittest
import numpy as np
import os
from PIL import Image, ImageDraw
from id_photo_utils import IdPhotoProcessor
import io

class TestIdPhotoProcessor(unittest.TestCase):
    def setUp(self):
        # 创建一个虚拟图像：100x100，黑色背景，中间有个白色圆形
        self.width = 100
        self.height = 100
        self.image = Image.new('RGB', (self.width, self.height), color='black')
        
        # 画一个白色圆形
        draw = ImageDraw.Draw(self.image)
        draw.ellipse((25, 25, 75, 75), fill='white')
        
        # 保存为 bytes 以模拟文件加载或直接传递 bytes
        img_byte_arr = io.BytesIO()
        self.image.save(img_byte_arr, format='PNG')
        self.img_bytes = img_byte_arr.getvalue()
        
        # 创建一个临时文件用于测试相对路径加载
        self.test_filename = "temp_test_image.png"
        self.image.save(self.test_filename)

    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_init_with_bytes(self):
        """测试使用 bytes 初始化"""
        processor = IdPhotoProcessor(image_data=self.img_bytes)
        self.assertIsNotNone(processor.original_image)
        self.assertEqual(processor.original_image.size, (100, 100))

    def test_init_with_relative_path(self):
        """测试使用相对路径文件初始化"""
        # 使用相对路径
        relative_path = f"./{self.test_filename}"
        processor = IdPhotoProcessor(image_path=relative_path)
        self.assertIsNotNone(processor.original_image)
        self.assertEqual(processor.original_image.size, (100, 100))

    def test_add_background_color(self):
        """测试更换背景颜色"""
        processor = IdPhotoProcessor(image_data=self.img_bytes)
        # 换成红底
        red_bg = (255, 0, 0)
        result = processor.add_background_color(red_bg)
        
        self.assertEqual(result.size, (100, 100))
        self.assertEqual(result.mode, 'RGB')
        
        # 检查结果是否为图像对象
        # 注意：由于 rembg 是深度学习模型，对合成图形的效果不确定，
        # 这里主要测试代码逻辑是否跑通，不校验具体像素值。
        
    def test_beautify(self):
        """测试美颜功能"""
        processor = IdPhotoProcessor(image_data=self.img_bytes)
        # 应用美颜
        result = processor.beautify(smooth_strength=10, brighten_strength=1.5)
        
        self.assertEqual(result.size, (100, 100))
        
        # 测试提亮效果
        # 创建一个灰色图像来测试
        gray_img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        img_byte_arr = io.BytesIO()
        gray_img.save(img_byte_arr, format='PNG')
        
        proc_gray = IdPhotoProcessor(image_data=img_byte_arr.getvalue())
        res_gray = proc_gray.beautify(smooth_strength=0, brighten_strength=1.5)
        
        # 中心像素应该比 100 亮
        center_pixel = res_gray.getpixel((50, 50))
        # 100 * 1.5 = 150. 应该在 150 左右。
        # 允许一定的浮点误差
        self.assertTrue(140 < center_pixel[0] < 160, f"预期像素值在 150 左右，实际得到 {center_pixel}")

if __name__ == '__main__':
    unittest.main()
