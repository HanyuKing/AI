import httpx
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any

class BingService:
    def __init__(self):
        from server.core.config import settings
        # 缓存文件存储在临时目录
        self.cache_dir = settings.TEMP_DIR / "bing_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "wallpaper_cache.json"
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._cache_date: Optional[str] = None

    async def get_wallpaper(self) -> Dict[str, str]:
        """
        Get Bing daily wallpaper URL and metadata.
        每天只请求一次，缓存到服务端文件。
        Returns a dict with 'url', 'copyright', 'title'.
        """
        today = date.today().strftime("%Y%m%d")
        
        # 检查内存缓存（同一天）
        if self._memory_cache and self._cache_date == today:
                return self._memory_cache

        # 检查文件缓存
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    cache_date = cache_data.get("date", "")
                    # 如果是今天的缓存，直接返回
                    if cache_date == today:
                        self._memory_cache = cache_data
                        self._cache_date = today
                        return cache_data
            except Exception as e:
                print(f"Error reading cache file: {e}")
        
        # 需要重新获取
        try:
            async with httpx.AsyncClient() as client:
                # Bing HPImageArchive API
                # format=js, idx=0 (today), n=1 (1 image), mkt=zh-CN
                response = await client.get(
                    "https://www.bing.com/HPImageArchive.aspx?format=js&idx=0&n=1&mkt=zh-CN",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "images" in data and len(data["images"]) > 0:
                        image = data["images"][0]
                        base_url = "https://www.bing.com"
                        url = base_url + image["url"]
                        
                        result = {
                            "url": url,
                            "copyright": image.get("copyright", ""),
                            "title": image.get("title", ""),
                            "date": today,  # 使用今天的日期作为缓存键
                            "startdate": image.get("startdate", "")
                        }
                        
                        # 保存到文件缓存
                        try:
                            with open(self.cache_file, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                        except Exception as e:
                            print(f"Error saving cache file: {e}")
                        
                        # 更新内存缓存
                        self._memory_cache = result
                        self._cache_date = today
                        return result
        except Exception as e:
            print(f"Error fetching Bing wallpaper: {e}")
            # 如果API失败，尝试返回旧的缓存（即使不是今天的）
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        old_cache = json.load(f)
                        # 移除date字段，返回旧数据
                        old_cache.pop("date", None)
                        return old_cache
                except Exception:
                    pass
            
        # Fallback if API fails
        return {
            "url": "", # Frontend should handle empty URL by showing default bg
            "copyright": "",
            "title": ""
        }



bing_service = BingService()
