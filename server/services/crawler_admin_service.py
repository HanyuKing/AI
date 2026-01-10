"""
爬虫管理服务
用于管理早好物爬虫的配置、Cookie、启动停止等
"""

import json
import subprocess
import os
import signal
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class CrawlerAdminService:
    """爬虫管理服务"""
    
    def __init__(self):
        # 获取项目根目录（server的父目录）
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.crawler_dir = self.base_dir / "scripts" / "zaohaowu"
        self.config_file = self.crawler_dir / "cookies.json"
        self.data_dir = self.crawler_dir / "data"
        self.pid_file = self.data_dir / "crawler.pid"
        self.log_file = self.crawler_dir / "crawler.log"
        
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果配置文件不存在，创建默认配置
        if not self.config_file.exists():
            default_config = {
                "cookies": [],
                "schedule": {
                    "enabled": True,
                    "start_hour": 7,
                    "end_hour": 9
                },
                "max_votes_per_day": 10
            }
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取爬虫配置"""
        if not self.config_file.exists():
            return {
                "cookies": [],
                "schedule": {
                    "enabled": True,
                    "start_hour": 7,
                    "end_hour": 9
                },
                "max_votes_per_day": 10
            }
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    def save_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """保存爬虫配置"""
        try:
            # 验证配置格式
            if "cookies" not in config:
                return {"success": False, "error": "缺少cookies字段"}
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return {"success": True, "message": "配置保存成功"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """获取爬虫运行状态"""
        is_running = False
        pid = None
        
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # 检查进程是否存在
                try:
                    os.kill(pid, 0)  # 发送0信号检查进程是否存在
                    is_running = True
                except OSError:
                    # 进程不存在，删除PID文件
                    self.pid_file.unlink()
                    pid = None
            except Exception:
                pass
        
        # 获取日志文件信息
        log_size = 0
        log_lines = []
        if self.log_file.exists():
            log_size = self.log_file.stat().st_size
            try:
                # 读取最后50行
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    log_lines = lines[-50:] if len(lines) > 50 else lines
            except Exception:
                pass
        
        # 获取投票统计
        vote_stats = self._get_vote_stats()
        
        return {
            "is_running": is_running,
            "pid": pid,
            "log_size": log_size,
            "log_lines": log_lines,
            "vote_stats": vote_stats
        }
    
    def _get_vote_stats(self) -> Dict[str, Any]:
        """获取投票统计信息"""
        stats = {
            "total_cookies": 0,
            "today_votes": {},
            "total_today_votes": 0
        }
        
        if not self.data_dir.exists():
            return stats
        
        today = datetime.now().strftime("%Y-%m-%d")
        vote_files = list(self.data_dir.glob("vote_count_*.json"))
        stats["total_cookies"] = len(vote_files)
        
        total_votes = 0
        for vote_file in vote_files:
            try:
                with open(vote_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        count = data.get("count", 0)
                        cookie_id = vote_file.stem.replace("vote_count_", "")
                        stats["today_votes"][cookie_id] = count
                        total_votes += count
            except Exception:
                pass
        
        stats["total_today_votes"] = total_votes
        return stats
    
    def get_cookies(self) -> List[Dict[str, Any]]:
        """获取所有Cookie列表（带投票统计）"""
        config = self.get_config()
        cookies = config.get("cookies", [])
        
        # 兼容旧格式（字符串数组）和新格式（对象数组）
        cookie_list = []
        for idx, cookie_item in enumerate(cookies):
            if isinstance(cookie_item, str):
                # 旧格式：字符串
                cookie_id = hashlib.md5(cookie_item.encode()).hexdigest()[:8]
                cookie_list.append({
                    "id": cookie_id,
                    "name": f"Cookie {idx + 1}",
                    "cookie": cookie_item,
                    "index": idx
                })
            elif isinstance(cookie_item, dict):
                # 新格式：对象
                cookie_str = cookie_item.get("cookie", "")
                cookie_id = hashlib.md5(cookie_str.encode()).hexdigest()[:8]
                cookie_list.append({
                    "id": cookie_id,
                    "name": cookie_item.get("name", f"Cookie {idx + 1}"),
                    "cookie": cookie_str,
                    "index": idx
                })
        
        # 获取每个Cookie的投票统计（只获取今日投票数，不加载完整日志以提高性能）
        today = datetime.now().strftime("%Y-%m-%d")
        for cookie_info in cookie_list:
            vote_file = self.data_dir / f"vote_count_{cookie_info['id']}.json"
            cookie_info["today_votes"] = 0
            # 不在这里加载完整日志，只标记是否有今日投票记录（用于前端判断是否显示"查看详情"按钮）
            cookie_info["has_today_logs"] = False
            
            if vote_file.exists():
                try:
                    with open(vote_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 获取今日投票数
                        if data.get("date") == today:
                            cookie_info["today_votes"] = data.get("count", 0)
                            # 检查是否有今日日志（只检查是否存在，不加载完整数据）
                            logs = data.get("logs", [])
                            cookie_info["has_today_logs"] = any(log.get("date") == today for log in logs)
                except Exception:
                    pass
        
        return cookie_list
    
    def add_cookie(self, name: str, cookie: str) -> Dict[str, Any]:
        """添加Cookie"""
        try:
            config = self.get_config()
            cookies = config.get("cookies", [])
            
            # 转换为新格式（如果还是旧格式）
            cookie_list = []
            for idx, item in enumerate(cookies):
                if isinstance(item, str):
                    cookie_list.append({
                        "name": f"Cookie {idx + 1}",
                        "cookie": item
                    })
                else:
                    cookie_list.append(item)
            
            # 添加新Cookie
            cookie_list.append({
                "name": name or f"Cookie {len(cookie_list) + 1}",
                "cookie": cookie
            })
            
            config["cookies"] = cookie_list
            return self.save_config(config)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def update_cookie(self, index: int, name: str, cookie: str) -> Dict[str, Any]:
        """更新Cookie"""
        try:
            config = self.get_config()
            cookies = config.get("cookies", [])
            
            if index < 0 or index >= len(cookies):
                return {"success": False, "error": "索引超出范围"}
            
            # 转换为新格式（如果还是旧格式）
            cookie_list = []
            for idx, item in enumerate(cookies):
                if isinstance(item, str):
                    cookie_list.append({
                        "name": f"Cookie {idx + 1}",
                        "cookie": item
                    })
                else:
                    cookie_list.append(item)
            
            # 更新Cookie
            cookie_list[index] = {
                "name": name or f"Cookie {index + 1}",
                "cookie": cookie
            }
            
            config["cookies"] = cookie_list
            return self.save_config(config)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_cookie(self, index: int) -> Dict[str, Any]:
        """删除Cookie"""
        try:
            config = self.get_config()
            cookies = config.get("cookies", [])
            
            if index < 0 or index >= len(cookies):
                return {"success": False, "error": "索引超出范围"}
            
            # 转换为新格式以获取cookie_id
            cookie_item = cookies[index]
            if isinstance(cookie_item, str):
                cookie_id = hashlib.md5(cookie_item.encode()).hexdigest()[:8]
            else:
                cookie_str = cookie_item.get("cookie", "")
                cookie_id = hashlib.md5(cookie_str.encode()).hexdigest()[:8]
            
            # 删除Cookie
            cookies.pop(index)
            config["cookies"] = cookies
            
            # 删除对应的投票日志文件
            vote_file = self.data_dir / f"vote_count_{cookie_id}.json"
            if vote_file.exists():
                vote_file.unlink()
            
            return self.save_config(config)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_cookie_vote_logs(self, cookie_id: str, days: int = 7) -> Dict[str, Any]:
        """获取指定Cookie的投票日志（以天为维度）"""
        vote_file = self.data_dir / f"vote_count_{cookie_id}.json"
        
        if not vote_file.exists():
            return {"success": False, "error": "Cookie不存在或没有投票记录"}
        
        try:
            with open(vote_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取投票日志
            logs = data.get("logs", [])
            
            # 按日期分组
            daily_logs = {}
            for log in logs:
                date = log.get("date", "")
                if date not in daily_logs:
                    daily_logs[date] = []
                daily_logs[date].append(log)
            
            # 获取最近N天的数据（按日期倒序）
            result_logs = []
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                day_logs = daily_logs.get(date, [])
                # 按时间戳倒序排列（最新的在前）
                day_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                result_logs.append({
                    "date": date,
                    "count": len(day_logs),
                    "logs": day_logs
                })
            
            # 按日期倒序排列（最新的日期在前）
            result_logs.sort(key=lambda x: x["date"], reverse=True)
            
            return {
                "success": True,
                "cookie_id": cookie_id,
                "logs": result_logs
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def start_crawler(self) -> Dict[str, Any]:
        """启动爬虫"""
        # 检查是否已在运行
        status = self.get_status()
        if status["is_running"]:
            return {"success": False, "error": "已在运行中"}
        
        try:
            # 使用nohup后台启动
            log_file = str(self.log_file)
            script_file = str(self.crawler_dir / "zaohaowu_crawler.py")
            
            # 启动命令
            cmd = [
                "python3", "-u", script_file
            ]
            
            # 使用subprocess启动
            process = subprocess.Popen(
                cmd,
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                cwd=str(self.crawler_dir),
                start_new_session=True
            )
            
            # 保存PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            return {
                "success": True,
                "message": "启动成功",
                "pid": process.pid
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def stop_crawler(self) -> Dict[str, Any]:
        """停止爬虫"""
        status = self.get_status()
        
        if not status["is_running"]:
            return {"success": False, "error": "未运行"}
        
        try:
            pid = status["pid"]
            # 发送SIGTERM信号
            os.kill(pid, signal.SIGTERM)
            
            # 等待进程结束（最多等待5秒）
            import time
            for _ in range(50):  # 50次，每次0.1秒
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    # 进程已结束
                    break
            else:
                # 如果还没结束，强制杀死
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            
            # 删除PID文件
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            return {"success": True, "message": "已停止"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_logs(self, lines: int = 100) -> Dict[str, Any]:
        """获取日志内容"""
        if not self.log_file.exists():
            return {"success": False, "error": "日志文件不存在"}
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            return {
                "success": True,
                "lines": log_lines,
                "total_lines": len(all_lines)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clear_logs(self) -> Dict[str, Any]:
        """清空日志"""
        try:
            if self.log_file.exists():
                self.log_file.unlink()
            return {"success": True, "message": "日志已清空"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# 单例
crawler_admin_service = CrawlerAdminService()

