"""
早好物爬虫脚本
功能：
1. 调用用户信息接口
2. 获取排行榜数据并提取 uniqueWantItId（从 wantItRespMap 中提取）
3. 批量发送 want 请求（GET /aigc/api/ticket/wantIt）

反爬虫措施：
- 随机User-Agent
- 随机延迟
- 请求头伪装
- Cookie轮换
"""

import time
import json
import random
import httpx
import threading
import sys
import hashlib
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置无缓冲输出（确保日志实时写入）
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# 线程安全的打印锁
_print_lock = threading.Lock()

# 投票计数锁（用于文件操作）
_vote_count_lock = threading.Lock()

def get_timestamp() -> str:
    """获取格式化的时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_print(*args, **kwargs):
    """
    带时间戳的打印函数，立即刷新输出
    
    Args:
        *args: 要打印的内容
        **kwargs: print函数的其他参数
    """
    timestamp = get_timestamp()
    if args:
        content = ' '.join(str(arg) for arg in args)
        print(f"[{timestamp}] {content}", **kwargs, flush=True)
    else:
        print(f"[{timestamp}]", **kwargs, flush=True)


def thread_safe_print(*args, **kwargs):
    """
    线程安全的打印函数，自动添加时间戳，立即刷新输出
    
    Args:
        *args: 要打印的内容
        **kwargs: print函数的其他参数
    """
    with _print_lock:
        log_print(*args, **kwargs)


# User-Agent 池，模拟不同浏览器
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]


class ZaoHaoWuCrawler:
    def __init__(self, cookie: str = "", cookies: Optional[List[str]] = None, thread_id: Optional[int] = None, max_votes_per_day: int = 10):
        """
        初始化爬虫
        
        Args:
            cookie: 单个 Cookie（已废弃，建议使用 cookies）
            cookies: Cookie 列表，支持轮换使用
            thread_id: 线程ID，用于标识不同的并发任务
            max_votes_per_day: 每天最大投票数（默认10次）
        """
        # 支持多个Cookie
        if cookies:
            self.cookies = cookies
            self.current_cookie_index = 0
        elif cookie:
            self.cookies = [cookie]
            self.current_cookie_index = 0
        else:
            self.cookies = []
            self.current_cookie_index = 0
        
        self.thread_id = thread_id
        self.thread_prefix = f"[线程{thread_id}]" if thread_id is not None else ""
        self.base_url = "https://zaohaowu.com"
        self.max_votes_per_day = max_votes_per_day
        self.script_dir = Path(__file__).parent
        # 创建data文件夹用于存储生成的文件
        self.data_dir = self.script_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.cookies_file = self.script_dir / "cookies.json"
        self.today = datetime.now().strftime("%Y-%m-%d")
        # 线程级别的余额不足标志（每个Cookie任务独立）
        self.insufficient_balance = False
        self._update_headers()
        
        # 创建HTTP客户端，使用连接池（每个线程独立的客户端）
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    def _get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(USER_AGENTS)
    
    def _get_current_cookie(self) -> str:
        """获取当前使用的Cookie"""
        if not self.cookies:
            return ""
        return self.cookies[self.current_cookie_index]
    
    def _get_vote_count_file(self) -> Path:
        """获取投票计数文件路径"""
        cookie = self._get_current_cookie()
        cookie_id = hashlib.md5(cookie.encode()).hexdigest()[:8]
        return self.data_dir / f"vote_count_{cookie_id}.json"
    
    def _get_today_vote_count(self) -> int:
        """获取今日投票次数"""
        vote_file = self._get_vote_count_file()
        with _vote_count_lock:
            if vote_file.exists():
                try:
                    with open(vote_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("date") == self.today:
                            return data.get("count", 0)
                except Exception:
                    pass
            return 0
    
    def _increment_vote_count(self, want_it_id: str = "") -> bool:
        """
        增加投票计数，返回是否成功（未超过限制）
        
        Args:
            want_it_id: 投票的wantItId（用于记录日志）
        """
        vote_file = self._get_vote_count_file()
        current_count = self._get_today_vote_count()
        
        if current_count >= self.max_votes_per_day:
            return False
        
        with _vote_count_lock:
            try:
                # 读取现有数据
                data = {
                    "date": self.today,
                    "count": current_count + 1,
                    "logs": []
                }
                
                if vote_file.exists():
                    try:
                        with open(vote_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            # 如果日期相同，保留现有日志
                            if existing_data.get("date") == self.today:
                                data["logs"] = existing_data.get("logs", [])
                            # 如果日期不同，清空日志（新的一天）
                    except Exception:
                        pass
                
                # 添加新的投票日志
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "want_it_id": want_it_id,
                    "date": self.today
                }
                data["logs"].append(log_entry)
                
                # 保存数据
                with open(vote_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                return True
            except Exception as e:
                log_print(f"⚠️  保存投票计数失败: {e}")
                return False
    
    def _extract_cookie_from_headers(self, response: httpx.Response) -> Optional[str]:
        """
        从响应头中提取Cookie
        
        Args:
            response: HTTP响应对象
            
        Returns:
            提取的Cookie字符串，如果没有则返回None
        """
        # 方法1: 使用httpx的cookies属性（推荐）
        if hasattr(response, 'cookies') and response.cookies:
            # 将Cookies对象转换为字符串格式
            cookie_parts = []
            for name, value in response.cookies.items():
                cookie_parts.append(f"{name}={value}")
            
            if cookie_parts:
                new_cookie = '; '.join(cookie_parts)
                return new_cookie
        
        # 方法2: 从响应头的 Set-Cookie 中提取cookie（备用）
        set_cookies = response.headers.get_list("Set-Cookie")
        if set_cookies:
            # 合并所有Set-Cookie头
            cookie_parts = []
            for set_cookie in set_cookies:
                # Set-Cookie格式: name=value; path=/; domain=...
                # 只提取 name=value 部分
                cookie_part = set_cookie.split(';')[0].strip()
                if cookie_part:
                    cookie_parts.append(cookie_part)
            
            if cookie_parts:
                # 合并所有cookie部分
                new_cookie = '; '.join(cookie_parts)
                return new_cookie
        
        return None
    
    def _merge_cookies(self, current_cookie: str, new_cookie: str) -> str:
        """
        合并两个Cookie字符串
        
        Args:
            current_cookie: 当前Cookie字符串
            new_cookie: 新的Cookie字符串（从响应头获取）
            
        Returns:
            合并后的Cookie字符串
        """
        if not current_cookie:
            return new_cookie
        if not new_cookie:
            return current_cookie
        
        # 将cookie字符串转换为字典
        def cookie_to_dict(cookie_str: str) -> Dict[str, str]:
            cookie_dict = {}
            for item in cookie_str.split(';'):
                item = item.strip()
                if '=' in item:
                    key, value = item.split('=', 1)
                    cookie_dict[key.strip()] = value.strip()
            return cookie_dict
        
        # 合并cookie字典
        current_dict = cookie_to_dict(current_cookie)
        new_dict = cookie_to_dict(new_cookie)
        
        # 新cookie覆盖旧cookie中相同的key
        merged_dict = {**current_dict, **new_dict}
        
        # 转换回字符串格式
        merged_cookie = '; '.join([f"{k}={v}" for k, v in merged_dict.items()])
        return merged_cookie
    
    def _update_cookie_from_response(self, response: httpx.Response) -> bool:
        """
        从响应头中提取Cookie并更新当前Cookie
        
        Args:
            response: HTTP响应对象
            
        Returns:
            是否成功更新Cookie
        """
        new_cookie = self._extract_cookie_from_headers(response)
        if not new_cookie:
            return False
        
        # 获取当前cookie
        current_cookie = self._get_current_cookie()
        
        # 合并cookie（新cookie会覆盖旧cookie中相同的key）
        merged_cookie = self._merge_cookies(current_cookie, new_cookie)
        
        # 如果合并后的cookie与当前cookie不同，则更新
        if merged_cookie != current_cookie:
            # 更新当前cookie
            if self.cookies:
                self.cookies[self.current_cookie_index] = merged_cookie
                self._update_headers()
                
                # 保存到配置文件
                self._save_cookies_to_file()
                
                thread_safe_print(f"{self.thread_prefix} 🔄 Cookie已从响应头更新并保存到配置文件")
                return True
        
        return False
    
    def _save_cookies_to_file(self):
        """保存Cookie列表到配置文件"""
        if not self.cookies_file.exists():
            return
        
        try:
            with open(self.cookies_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 更新cookies列表
            config["cookies"] = self.cookies
            
            # 保存回文件
            with open(self.cookies_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            thread_safe_print(f"{self.thread_prefix} ⚠️  保存Cookie到配置文件失败: {e}")
    
    def _rotate_cookie(self):
        """轮换到下一个Cookie"""
        if len(self.cookies) > 1:
            self.current_cookie_index = (self.current_cookie_index + 1) % len(self.cookies)
            self._update_headers()
            thread_safe_print(f"{self.thread_prefix} 🔄 切换到 Cookie #{self.current_cookie_index + 1}/{len(self.cookies)}")
    
    def _update_headers(self):
        """更新请求头（包括随机User-Agent）"""
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Cookie": self._get_current_cookie(),
            "Referer": "https://zaohaowu.com/",
            "Origin": "https://zaohaowu.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
        }
    
    def _random_delay(self, base_delay: float, variance: float = 0.3):
        """
        随机延迟，模拟人类行为
        
        Args:
            base_delay: 基础延迟（秒）
            variance: 延迟变化幅度（0-1之间）
        """
        # 在基础延迟上增加随机变化
        min_delay = base_delay * (1 - variance)
        max_delay = base_delay * (1 + variance)
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        获取用户信息，并从响应头中提取并更新Cookie
        
        Returns:
            用户信息响应数据
        """
        url = f"{self.base_url}/aigc/api/user/info"
        params = {"_timer": int(time.time() * 1000)}
        
        try:
            # 每次请求前更新请求头（随机User-Agent）
            self._update_headers()
            response = self.client.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            
            # 从响应头中提取并更新Cookie
            self._update_cookie_from_response(response)
            
            # 检查是否需要切换Cookie
            if result.get("code") == 401:
                thread_safe_print(f"{self.thread_prefix} ⚠️  Cookie可能已过期，尝试切换Cookie...")
                self._rotate_cookie()
                # 重试一次
                self._update_headers()
                response = self.client.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                result = response.json()
                # 再次尝试更新Cookie
                self._update_cookie_from_response(response)
            
            thread_safe_print(f"{self.thread_prefix} ✓ 用户信息接口调用成功")
            return result
        except Exception as e:
            thread_safe_print(f"{self.thread_prefix} ✗ 用户信息接口调用失败: {e}")
            raise
    
    def get_ranking_want(self, limit: int = 40, want_it_ranking_type: int = 2, page: int = 1) -> List[Dict[str, Any]]:
        """
        获取排行榜数据（支持分页）
        
        Args:
            limit: 每页返回数量限制
            want_it_ranking_type: 排行榜类型
            page: 页码（从1开始）
            
        Returns:
            排行榜数据列表
        """
        url = f"{self.base_url}/aigc/api/ranking/want"
        params = {
            "limit": limit,
            "wantItRankingType": want_it_ranking_type,
            "page": page
        }
        
        try:
            # 每次请求前更新请求头（随机User-Agent）
            self._update_headers()
            response = self.client.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            
            # 检查是否需要切换Cookie
            if result.get("code") == 401:
                thread_safe_print(f"{self.thread_prefix} ⚠️  Cookie可能已过期，尝试切换Cookie...")
                self._rotate_cookie()
                # 重试一次
                self._update_headers()
                response = self.client.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                result = response.json()
            
            # 提取数据列表：数据结构为 {code, msg, data: {dataList: [...]}}
            if isinstance(result, dict) and result.get("code") == 200:
                data = result.get("data", {}).get("dataList", [])
            else:
                data = []
            
            thread_safe_print(f"{self.thread_prefix} ✓ 排行榜接口调用成功（第{page}页），获取到 {len(data)} 条数据")
            return data
        except Exception as e:
            thread_safe_print(f"{self.thread_prefix} ✗ 排行榜接口调用失败: {e}")
            raise
    
    def get_ranking_want_multi_pages(self, pages: int = 2, limit: int = 40, want_it_ranking_type: int = 2) -> List[Dict[str, Any]]:
        """
        获取多页排行榜数据并合并
        
        Args:
            pages: 获取的页数（默认2页）
            limit: 每页返回数量限制
            want_it_ranking_type: 排行榜类型
            
        Returns:
            合并后的排行榜数据列表
        """
        all_data = []
        for page in range(1, pages + 1):
            try:
                page_data = self.get_ranking_want(limit=limit, want_it_ranking_type=want_it_ranking_type, page=page)
                all_data.extend(page_data)
                # 页面之间添加延迟，模拟人类操作
                if page < pages:
                    self._random_delay(1.0, variance=0.3)
            except Exception as e:
                thread_safe_print(f"{self.thread_prefix} ⚠️  获取第{page}页数据失败: {e}")
                # 如果某一页失败，继续获取下一页
                continue
        
        thread_safe_print(f"{self.thread_prefix} ✓ 共获取 {pages} 页数据，合并后共 {len(all_data)} 条")
        return all_data
    
    def extract_unique_want_it_ids(self, ranking_data: List[Dict[str, Any]]) -> List[str]:
        """
        从排行榜数据中提取 uniqueWantItId（从 wantItRespMap 中提取）
        
        Args:
            ranking_data: 排行榜数据列表（dataList）
            
        Returns:
            uniqueWantItId 列表
        """
        want_it_ids = []
        for item in ranking_data:
            # 从 wantItRespMap 中提取 uniqueWantItId
            want_it_resp_map = item.get("wantItRespMap", {})
            if isinstance(want_it_resp_map, dict):
                for key, value in want_it_resp_map.items():
                    if isinstance(value, dict):
                        unique_want_it_id = value.get("uniqueWantItId")
                        if unique_want_it_id:
                            want_it_ids.append(unique_want_it_id)
        
        thread_safe_print(f"{self.thread_prefix} ✓ 提取到 {len(want_it_ids)} 个 uniqueWantItId")
        return want_it_ids
    
    def _check_insufficient_balance(self, result: Dict[str, Any]) -> bool:
        """
        检查响应中是否包含余额不足的错误
        
        Args:
            result: 响应数据
            
        Returns:
            是否余额不足
        """
        msg = result.get("msg", "").lower() if result.get("msg") else ""
        error = result.get("error", "").lower() if result.get("error") else ""
        
        # 检查是否包含余额不足相关的关键词
        balance_keywords = ["余额不足", "insufficient", "balance", "余额", "不足"]
        for keyword in balance_keywords:
            if keyword in msg or keyword in error:
                return True
        return False
    
    def send_want_request(self, want_it_id: str) -> Dict[str, Any]:
        """
        发送 want 请求（带每日投票限制检查）
        
        Args:
            want_it_id: uniqueWantItId
            
        Returns:
            请求响应数据
        """
        # 检查是否已标记余额不足（线程级别）
        if self.insufficient_balance:
            thread_safe_print(f"{self.thread_prefix} ⚠️  检测到余额不足，停止执行")
            return {"success": False, "error": "余额不足", "code": 402, "insufficient_balance": True}
        
        # 检查今日投票限制
        current_count = self._get_today_vote_count()
        if current_count >= self.max_votes_per_day:
            thread_safe_print(f"{self.thread_prefix} ⚠️  今日投票次数已达上限（{current_count}/{self.max_votes_per_day}），跳过")
            return {"success": False, "error": "已达每日投票上限", "code": 429, "skipped": True}
        
        url = f"{self.base_url}/aigc/api/ticket/wantIt"
        params = {
            "wantItId": want_it_id,
            "_timer": int(time.time() * 1000)
        }
        
        try:
            # 每次请求前更新请求头（随机User-Agent）
            self._update_headers()
            response = self.client.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            
            # 检查余额不足（线程级别）
            if self._check_insufficient_balance(result):
                self.insufficient_balance = True
                thread_safe_print(f"{self.thread_prefix} ⚠️  余额不足，立即停止当前线程执行")
                return {"success": False, "error": "余额不足", "code": result.get("code", 402), "insufficient_balance": True}
            
            # 如果返回401，尝试切换Cookie并重试
            if result.get("code") == 401:
                thread_safe_print(f"{self.thread_prefix} ⚠️  Cookie可能已过期，尝试切换Cookie...")
                self._rotate_cookie()
                self._update_headers()
                response = self.client.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                result = response.json()
                
                # 再次检查余额不足（线程级别）
                if self._check_insufficient_balance(result):
                    self.insufficient_balance = True
                    thread_safe_print(f"{self.thread_prefix} ⚠️  余额不足，立即停止当前线程执行")
                    return {"success": False, "error": "余额不足", "code": result.get("code", 402), "insufficient_balance": True}
            
            # 投票成功，增加计数
            if result.get("code") == 200 or result.get("success") is not False:
                if self._increment_vote_count(want_it_id):
                    new_count = self._get_today_vote_count()
                    remaining = self.max_votes_per_day - new_count
                    thread_safe_print(f"{self.thread_prefix}   今日已投票: {new_count}/{self.max_votes_per_day}, 剩余: {remaining}")
            
            return result
        except httpx.HTTPStatusError as e:
            # 处理 HTTP 错误（如 401 登录过期）
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("msg", error_msg)
            except:
                pass
            thread_safe_print(f"{self.thread_prefix} ✗ 请求失败 (wantItId: {want_it_id}): {error_msg}")
            return {"success": False, "error": error_msg, "code": e.response.status_code}
        except Exception as e:
            thread_safe_print(f"{self.thread_prefix} ✗ 请求异常 (wantItId: {want_it_id}): {e}")
            return {"success": False, "error": str(e)}
    
    def run(self, limit: int = 40, want_it_ranking_type: int = 2, delay: float = 0.5) -> Dict[str, Any]:
        """
        执行完整的爬虫流程
        
        Args:
            limit: 排行榜返回数量限制
            want_it_ranking_type: 排行榜类型
            delay: 每次请求之间的延迟（秒）
            
        Returns:
            执行结果统计
        """
        thread_safe_print(f"{self.thread_prefix} " + "=" * 50)
        thread_safe_print(f"{self.thread_prefix} 早好物爬虫开始运行")
        thread_safe_print(f"{self.thread_prefix} " + "=" * 50)
        
        # 步骤1: 调用用户信息接口
        thread_safe_print(f"{self.thread_prefix} \n[步骤 1] 调用用户信息接口...")
        try:
            self.get_user_info()
        except Exception as e:
            thread_safe_print(f"{self.thread_prefix} 警告: 用户信息接口调用失败，继续执行: {e}")
        
        # 步骤2: 等待（使用随机延迟，模拟人类操作）
        thread_safe_print(f"{self.thread_prefix} \n[步骤 2] 等待中...")
        self._random_delay(2.0, variance=0.3)  # 2秒 ± 30%，模拟人类操作间隔
        
        # 步骤3: 获取排行榜数据（获取2页，防作弊）
        thread_safe_print(f"{self.thread_prefix} \n[步骤 3] 获取排行榜数据（获取2页）...")
        ranking_data = self.get_ranking_want_multi_pages(pages=10, limit=limit, want_it_ranking_type=want_it_ranking_type)
        
        if not ranking_data:
            thread_safe_print(f"{self.thread_prefix} ✗ 未获取到排行榜数据，退出")
            return {"total": 0, "success": 0, "fail": 0}
        
        # 步骤4: 提取 uniqueWantItId
        thread_safe_print(f"{self.thread_prefix} \n[步骤 4] 提取 uniqueWantItId...")
        want_it_ids = self.extract_unique_want_it_ids(ranking_data)
        
        if not want_it_ids:
            thread_safe_print(f"{self.thread_prefix} ✗ 未提取到任何 wantItId，退出")
            return {"total": 0, "success": 0, "fail": 0}
        
        # 步骤4.5: 随机打乱want列表（防作弊）
        random.shuffle(want_it_ids)
        thread_safe_print(f"{self.thread_prefix} ✓ 已随机打乱 want 列表，共 {len(want_it_ids)} 个")
        
        # 步骤5: 批量发送 want 请求
        current_vote_count = self._get_today_vote_count()
        remaining_votes = max(0, self.max_votes_per_day - current_vote_count)
        
        thread_safe_print(f"{self.thread_prefix} \n[步骤 5] 开始批量发送 want 请求")
        thread_safe_print(f"{self.thread_prefix} 今日已投票: {current_vote_count}/{self.max_votes_per_day}, 剩余可投票: {remaining_votes} 次")
        thread_safe_print(f"{self.thread_prefix} 待处理: {len(want_it_ids)} 个")
        
        if remaining_votes <= 0:
            thread_safe_print(f"{self.thread_prefix} ⚠️  今日投票次数已达上限，跳过所有请求")
            return {
                "total": len(want_it_ids),
                "success": 0,
                "fail": 0,
                "skipped": len(want_it_ids)
            }
        
        # 限制处理数量不超过剩余可投票数
        want_it_ids_to_process = want_it_ids[:remaining_votes]
        skipped_count = len(want_it_ids) - len(want_it_ids_to_process)
        
        if skipped_count > 0:
            thread_safe_print(f"{self.thread_prefix} ⚠️  将跳过 {skipped_count} 个请求（超过每日限制）")
        
        success_count = 0
        fail_count = 0
        skipped_count_actual = 0
        
        for i, want_it_id in enumerate(want_it_ids_to_process, 1):
            # 检查是否余额不足（线程级别）
            if self.insufficient_balance:
                thread_safe_print(f"{self.thread_prefix} ⚠️  检测到余额不足，立即停止当前线程执行")
                break
            
            thread_safe_print(f"{self.thread_prefix} \n[{i}/{len(want_it_ids_to_process)}] 处理 wantItId: {want_it_id}")
            result = self.send_want_request(want_it_id)
            
            # 检查余额不足
            if result.get("insufficient_balance"):
                thread_safe_print(f"{self.thread_prefix}   ⚠️  余额不足，立即停止执行")
                break
            
            if result.get("skipped"):
                skipped_count_actual += 1
                thread_safe_print(f"{self.thread_prefix}   ⏭️  跳过（已达每日限制）")
            elif result.get("success") is not False and result.get("code") != 401:
                success_count += 1
                thread_safe_print(f"{self.thread_prefix}   ✓ 成功")
            else:
                fail_count += 1
                thread_safe_print(f"{self.thread_prefix}   ✗ 失败: {result.get('msg', result.get('error', '未知错误'))}")
            
            # 随机延迟，避免请求过快（模拟人类行为）
            if i < len(want_it_ids_to_process):
                # 在基础延迟上增加随机变化，模拟人类操作的不规律性
                self._random_delay(delay, variance=0.4)
        
        # 统计结果
        final_vote_count = self._get_today_vote_count()
        thread_safe_print(f"{self.thread_prefix} \n" + "=" * 50)
        thread_safe_print(f"{self.thread_prefix} 爬虫执行完成")
        thread_safe_print(f"{self.thread_prefix} " + "=" * 50)
        thread_safe_print(f"{self.thread_prefix} 总计: {len(want_it_ids)} 个")
        thread_safe_print(f"{self.thread_prefix} 成功: {success_count} 个")
        thread_safe_print(f"{self.thread_prefix} 失败: {fail_count} 个")
        if skipped_count > 0 or skipped_count_actual > 0:
            thread_safe_print(f"{self.thread_prefix} 跳过: {skipped_count + skipped_count_actual} 个（超过每日限制）")
        thread_safe_print(f"{self.thread_prefix} 今日总投票: {final_vote_count}/{self.max_votes_per_day}")
        if self.insufficient_balance:
            thread_safe_print(f"{self.thread_prefix} ⚠️  余额不足，当前线程已停止执行")
        thread_safe_print(f"{self.thread_prefix} " + "=" * 50)
        
        return {
            "total": len(want_it_ids),
            "success": success_count,
            "fail": fail_count,
            "skipped": skipped_count + skipped_count_actual,
            "insufficient_balance": self.insufficient_balance
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


def load_config_from_file(file_path: str = "cookies.json") -> Dict[str, Any]:
    """
    从配置文件加载配置（包括Cookie和定时任务设置）
    
    Args:
        file_path: 配置文件路径（相对于脚本目录）
        
    Returns:
        配置字典，包含 cookies 和 schedule
    """
    script_dir = Path(__file__).parent
    config_file = script_dir / file_path
    
    if not config_file.exists():
        log_print(f"⚠️  配置文件不存在: {config_file}")
        return {"cookies": [], "schedule": {"enabled": True, "start_hour": 7, "end_hour": 9}}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 提取Cookie列表
            cookies = data.get("cookies", [])
            if cookies:
                log_print(f"✓ 从配置文件加载了 {len(cookies)} 个Cookie")
            else:
                log_print("⚠️  配置文件中没有找到Cookie")
            
            # 提取定时任务配置
            schedule = data.get("schedule", {})
            schedule_enabled = schedule.get("enabled", True)
            schedule_start_hour = schedule.get("start_hour", 7)
            schedule_end_hour = schedule.get("end_hour", 9)
            
            log_print(f"✓ 定时任务配置: enabled={schedule_enabled}, 运行时间={schedule_start_hour}:00-{schedule_end_hour}:00")
            
            # 提取投票限制配置
            max_votes_per_day = data.get("max_votes_per_day", 10)
            log_print(f"✓ 每日投票限制: 每个Cookie最多 {max_votes_per_day} 次")
            
            return {
                "cookies": cookies,
                "schedule": {
                    "enabled": schedule_enabled,
                    "start_hour": schedule_start_hour,
                    "end_hour": schedule_end_hour
                },
                "max_votes_per_day": max_votes_per_day
            }
    except json.JSONDecodeError as e:
        log_print(f"✗ 配置文件格式错误: {e}")
        return {"cookies": [], "schedule": {"enabled": True, "start_hour": 7, "end_hour": 9}, "max_votes_per_day": 10}
    except Exception as e:
        log_print(f"✗ 读取配置文件失败: {e}")
        return {"cookies": [], "schedule": {"enabled": True, "start_hour": 7, "end_hour": 9}, "max_votes_per_day": 10}


def load_cookies_from_file(file_path: str = "cookies.json") -> List[str]:
    """
    从配置文件加载Cookie列表（兼容旧接口）
    支持旧格式（字符串数组）和新格式（对象数组）
    
    Args:
        file_path: Cookie配置文件路径（相对于脚本目录）
        
    Returns:
        Cookie字符串列表
    """
    config = load_config_from_file(file_path)
    cookies = config.get("cookies", [])
    
    # 转换为字符串列表（兼容新旧格式）
    cookie_list = []
    for item in cookies:
        if isinstance(item, str):
            # 旧格式：直接是字符串
            cookie_list.append(item)
        elif isinstance(item, dict):
            # 新格式：对象，提取cookie字段
            cookie_list.append(item.get("cookie", ""))
    
    return cookie_list


def run_single_cookie_loop(cookie: str, thread_id: int, limit: int = 40, want_it_ranking_type: int = 2, delay: float = 0.5, max_votes_per_day: int = 10, start_hour: int = 7, end_hour: int = 9, schedule_enabled: bool = True):
    """
    使用单个Cookie运行爬虫（独立循环，每个线程独立控制时间）
    
    Args:
        cookie: 单个Cookie
        thread_id: 线程ID
        limit: 排行榜返回数量限制
        want_it_ranking_type: 排行榜类型
        delay: 每次请求之间的延迟（秒）
        max_votes_per_day: 每天最大投票数（默认10次）
        start_hour: 运行开始时间（24小时制）
        end_hour: 运行结束时间（24小时制）
        schedule_enabled: 是否启用定时任务
    """
    thread_safe_print(f"[线程{thread_id}] 🚀 线程启动，开始独立循环执行")
    
    # 首次启动立即执行
    first_run = True
    
    while True:
        try:
            # 如果不是首次运行且启用了定时任务，需要等待到时间范围内
            if not first_run and schedule_enabled:
                now = datetime.now()
                # 检查当前时间是否在运行时间范围内
                if not is_in_time_range(start_hour, end_hour):
                    thread_safe_print(f"[线程{thread_id}] ⏰ 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    thread_safe_print(f"[线程{thread_id}] ⏰ 不在运行时间范围内（{start_hour}:00 - {end_hour}:00）")
                    
                    # 生成随机运行时间（在指定时间范围内）
                    random_time = get_random_time_in_range(start_hour, end_hour)
                    thread_safe_print(f"[线程{thread_id}] ⏰ 将在 {random_time.strftime('%Y-%m-%d %H:%M:%S')} 随机运行")
                    
                    # 等待到随机时间
                    wait_seconds = (random_time - now).total_seconds()
                    if wait_seconds > 0:
                        wait_hours = wait_seconds / 3600
                        thread_safe_print(f"[线程{thread_id}] ⏰ 等待时间: {wait_hours:.2f} 小时 ({wait_seconds:.0f} 秒)")
                        thread_safe_print(f"[线程{thread_id}] ⏰ 等待中...")
                        time.sleep(wait_seconds)
                        thread_safe_print(f"[线程{thread_id}] ✓ 到达目标时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    thread_safe_print(f"[线程{thread_id}] ⏰ 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    thread_safe_print(f"[线程{thread_id}] ⏰ 在运行时间范围内（{start_hour}:00 - {end_hour}:00）")
            
            # 执行爬虫任务
            with ZaoHaoWuCrawler(cookie=cookie, thread_id=thread_id, max_votes_per_day=max_votes_per_day) as crawler:
                result = crawler.run(
                    limit=limit,
                    want_it_ranking_type=want_it_ranking_type,
                    delay=delay
                )
                
                # 检查是否投票完成或余额不足
                insufficient_balance = result.get("insufficient_balance", False)
                final_vote_count = crawler._get_today_vote_count()
                votes_completed = final_vote_count >= max_votes_per_day
                
                if insufficient_balance:
                    thread_safe_print(f"[线程{thread_id}] ⚠️  余额不足，线程停止，等待下一次任务执行")
                elif votes_completed:
                    thread_safe_print(f"[线程{thread_id}] ✓ 今日投票已完成（{final_vote_count}/{max_votes_per_day}），线程停止，等待下一次任务执行")
                else:
                    thread_safe_print(f"[线程{thread_id}] ✓ 本次执行完成，等待下一次任务执行")
                
                # 如果启用了定时任务，等待到下一个时间窗口
                if schedule_enabled:
                    # 生成下一个随机执行时间（在时间范围内）
                    now = datetime.now()
                    random_time = get_random_time_in_range(start_hour, end_hour)
                    
                    # 如果随机时间已过，设置为明天
                    if random_time <= now:
                        random_time += timedelta(days=1)
                    
                    wait_seconds = (random_time - now).total_seconds()
                    wait_hours = wait_seconds / 3600
                    thread_safe_print(f"[线程{thread_id}] ⏰ 下次执行时间: {random_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    thread_safe_print(f"[线程{thread_id}] ⏰ 等待时间: {wait_hours:.2f} 小时 ({wait_seconds:.0f} 秒)")
                    thread_safe_print(f"[线程{thread_id}] ⏰ 等待中...")
                    time.sleep(wait_seconds)
                else:
                    # 未启用定时任务，等待1小时后重试
                    thread_safe_print(f"[线程{thread_id}] ⏰ 定时任务未启用，等待1小时后重试...")
                    time.sleep(3600)
            
            first_run = False
            
        except Exception as e:
            thread_safe_print(f"[线程{thread_id}] ✗ 执行异常: {e}")
            thread_safe_print(f"[线程{thread_id}] ⏰ 等待1小时后重试...")
            time.sleep(3600)
            first_run = False


def run_single_cookie(cookie: str, thread_id: int, limit: int = 40, want_it_ranking_type: int = 2, delay: float = 0.5, max_votes_per_day: int = 10) -> Dict[str, Any]:
    """
    使用单个Cookie运行爬虫（单次执行，用于向后兼容）
    
    Args:
        cookie: 单个Cookie
        thread_id: 线程ID
        limit: 排行榜返回数量限制
        want_it_ranking_type: 排行榜类型
        delay: 每次请求之间的延迟（秒）
        max_votes_per_day: 每天最大投票数（默认10次）
        
    Returns:
        执行结果统计
    """
    try:
        with ZaoHaoWuCrawler(cookie=cookie, thread_id=thread_id, max_votes_per_day=max_votes_per_day) as crawler:
            result = crawler.run(
                limit=limit,
                want_it_ranking_type=want_it_ranking_type,
                delay=delay
            )
            return result
    except Exception as e:
        thread_safe_print(f"[线程{thread_id}] ✗ 执行异常: {e}")
        return {"total": 0, "success": 0, "fail": 0, "error": str(e)}


def wait_until_time(target_hour: int, target_minute: int = 0):
    """
    等待到指定时间
    
    Args:
        target_hour: 目标小时（0-23）
        target_minute: 目标分钟（0-59）
    """
    now = datetime.now()
    target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # 如果目标时间已过，则设置为明天
    if target_time <= now:
        target_time += timedelta(days=1)
    
    wait_seconds = (target_time - now).total_seconds()
    wait_hours = wait_seconds / 3600
    
    log_print(f"⏰ 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"⏰ 目标时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"⏰ 等待时间: {wait_hours:.2f} 小时 ({wait_seconds:.0f} 秒)")
    log_print("⏰ 等待中...")
    
    time.sleep(wait_seconds)
    log_print(f"✓ 到达目标时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def get_random_time_in_range(start_hour: int, end_hour: int) -> datetime:
    """
    在指定时间范围内生成随机时间
    
    Args:
        start_hour: 开始小时（0-23）
        end_hour: 结束小时（0-23）
        
    Returns:
        随机时间（datetime对象）
    """
    now = datetime.now()
    
    # 生成随机的小时和分钟
    random_hour = random.randint(start_hour, end_hour - 1)
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    
    target_time = now.replace(hour=random_hour, minute=random_minute, second=random_second, microsecond=0)
    
    # 如果时间已过，设置为明天
    if target_time <= now:
        target_time += timedelta(days=1)
    
    return target_time


def is_in_time_range(start_hour: int, end_hour: int) -> bool:
    """
    检查当前时间是否在指定范围内
    
    Args:
        start_hour: 开始小时（0-23）
        end_hour: 结束小时（0-23）
        
    Returns:
        是否在时间范围内
    """
    now = datetime.now()
    current_hour = now.hour
    return start_hour <= current_hour < end_hour


def main(schedule_mode: Optional[bool] = None, start_hour: Optional[int] = None, end_hour: Optional[int] = None):
    """
    主函数
    Cookie和定时任务配置从 cookies.json 文件中读取
    支持多Cookie并发执行，互不影响
    第一次启动时立即执行，之后才根据时间控制
    
    Args:
        schedule_mode: 是否启用定时任务模式（None表示从配置文件读取）
        start_hour: 定时任务开始小时（None表示从配置文件读取）
        end_hour: 定时任务结束小时（None表示从配置文件读取）
    """
    # 从配置文件加载配置
    config = load_config_from_file("cookies.json")
    cookies = config.get("cookies", [])
    schedule_config = config.get("schedule", {})
    max_votes_per_day = config.get("max_votes_per_day", 10)
    
    # 使用配置文件的值，如果参数提供了则使用参数值（参数优先级更高）
    schedule_enabled = schedule_mode if schedule_mode is not None else schedule_config.get("enabled", True)
    schedule_start_hour = start_hour if start_hour is not None else schedule_config.get("start_hour", 7)
    schedule_end_hour = end_hour if end_hour is not None else schedule_config.get("end_hour", 9)
    
    # 检查是否是第一次执行（使用文件记录执行状态）
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    first_run_flag_file = data_dir / ".first_run_completed"
    is_first_run = not first_run_flag_file.exists()
    
    # 程序启动时总是立即执行一次
    log_print("\n🚀 程序启动，立即执行")
    log_print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 定时任务模式（仅用于后续的循环执行，不影响首次启动）
    if schedule_enabled:
        log_print("=" * 60)
        log_print("定时任务模式已启用")
        log_print(f"运行时间范围: {schedule_start_hour}:00 - {schedule_end_hour}:00")
        log_print("（首次启动立即执行，后续将按时间控制）")
        log_print("=" * 60)
    
    # 使用已加载的Cookie（已在函数开头加载）
    
    if not cookies:
        log_print("\n⚠️  警告: 未找到有效的Cookie！")
        log_print("请创建 cookies.json 文件并添加Cookie，格式如下：")
        log_print("""
{
  "cookies": [
    "你的Cookie1",
    "你的Cookie2"
  ]
}
        """)
        return
    
    # 转换为字符串列表（兼容新旧格式）
    cookie_strings = []
    for item in cookies:
        if isinstance(item, str):
            cookie_strings.append(item)
        elif isinstance(item, dict):
            cookie_strings.append(item.get("cookie", ""))
    
    cookie_strings = [c for c in cookie_strings if c]  # 过滤空字符串
    
    if not cookie_strings:
        log_print("\n⚠️  警告: 未找到有效的Cookie！")
        log_print("请创建 cookies.json 文件并添加Cookie，格式如下：")
        log_print("""
{
  "cookies": [
    "你的Cookie1",
    "你的Cookie2"
  ]
}
        """)
        return
    
    log_print(f"\n✓ 加载了 {len(cookie_strings)} 个Cookie")
    log_print(f"✓ 将使用 {len(cookie_strings)} 个线程并发执行，每个线程独立控制时间")
    log_print(f"✓ 每日投票限制: 每个Cookie最多 {max_votes_per_day} 次")
    if schedule_enabled:
        log_print(f"✓ 运行时间范围: {schedule_start_hour}:00 - {schedule_end_hour}:00（每个线程独立随机）")
    log_print("")
    
    # 配置参数
    limit = 40
    want_it_ranking_type = 2
    # 增加延迟时间，模拟真实人类操作速度（2-4秒之间随机）
    delay = random.uniform(2.0, 4.0)  # 每次请求间隔2-4秒，更符合人类操作速度
    
    # 使用线程池并发执行，每个线程独立循环
    with ThreadPoolExecutor(max_workers=len(cookie_strings)) as executor:
        # 提交所有任务（每个线程独立循环）
        futures = []
        for idx, cookie in enumerate(cookie_strings):
            future = executor.submit(
                run_single_cookie_loop,
                cookie,
                idx + 1,
                limit,
                want_it_ranking_type,
                delay,
                max_votes_per_day,
                schedule_start_hour,
                schedule_end_hour,
                schedule_enabled
            )
            futures.append(future)
        
        log_print(f"✓ 已启动 {len(futures)} 个线程，每个线程独立循环执行")
        log_print("✓ 每个线程在投票完成或余额不足时会停止，等待下一次任务执行")
        log_print("")
        
        # 等待所有线程（实际上会一直运行）
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log_print(f"⚠️  线程执行异常: {e}")
        except KeyboardInterrupt:
            log_print("\n⚠️  收到中断信号，正在停止所有线程...")
            executor.shutdown(wait=False)


if __name__ == "__main__":
    # 从配置文件读取定时任务设置
    # 如需覆盖配置，可以传入参数：
    # main(schedule_mode=False)  # 立即执行，忽略配置文件
    # main(start_hour=8, end_hour=10)  # 覆盖运行时间范围
    main()

