import httpx
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class BingService:
    def __init__(self):
        self.cache_file = "bing_wallpaper_cache.json"
        self.cache_duration = timedelta(hours=12)
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None

    async def get_wallpaper(self) -> Dict[str, str]:
        """
        Get Bing daily wallpaper URL and metadata.
        Returns a dict with 'url', 'copyright', 'title'.
        """
        # Check memory cache first
        if self._memory_cache and self._last_fetch:
            if datetime.now() - self._last_fetch < self.cache_duration:
                return self._memory_cache

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
                        # High res URL usually available by replacing 1920x1080 with UHD if needed, 
                        # but standard is usually fine.
                        
                        result = {
                            "url": url,
                            "copyright": image.get("copyright", ""),
                            "title": image.get("title", ""),
                            "date": image.get("startdate", "")
                        }
                        
                        # Update cache
                        self._memory_cache = result
                        self._last_fetch = datetime.now()
                        return result
        except Exception as e:
            print(f"Error fetching Bing wallpaper: {e}")
            
        # Fallback if API fails
        return {
            "url": "", # Frontend should handle empty URL by showing default bg
            "copyright": "",
            "title": ""
        }



bing_service = BingService()
