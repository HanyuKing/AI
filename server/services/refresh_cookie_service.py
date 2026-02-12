"""
Cookie 刷新脚本管理服务
用于启动/停止/重启 refresh_cookie.py
"""

import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any


class RefreshCookieService:
    """Cookie 刷新脚本管理服务"""

    def __init__(self):
        # 项目根目录（server 的父目录）
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.script_dir = self.base_dir / "scripts" / "zaohaowu"
        self.data_dir = self.script_dir / "data"
        self.pid_file = self.data_dir / "refresh_cookie.pid"
        self.log_file = self.script_dir / "refresh_cookie.log"

        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_status(self) -> Dict[str, Any]:
        """获取脚本运行状态"""
        is_running = False
        pid = None

        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    pid = int(f.read().strip())

                try:
                    os.kill(pid, 0)
                    is_running = True
                except OSError:
                    self.pid_file.unlink()
                    pid = None
            except Exception:
                pass

        return {
            "is_running": is_running,
            "pid": pid
        }

    def start(self) -> Dict[str, Any]:
        """启动刷新脚本"""
        status = self.get_status()
        if status["is_running"]:
            return {"success": False, "error": "已在运行中"}

        try:
            script_file = str(self.script_dir / "refresh_cookie.py")
            cmd = ["python3", "-u", script_file]

            process = subprocess.Popen(
                cmd,
                stdout=open(self.log_file, "a"),
                stderr=subprocess.STDOUT,
                cwd=str(self.script_dir),
                start_new_session=True
            )

            with open(self.pid_file, "w") as f:
                f.write(str(process.pid))

            return {"success": True, "message": "启动成功", "pid": process.pid}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """停止刷新脚本"""
        status = self.get_status()
        if not status["is_running"]:
            return {"success": False, "error": "未运行"}

        try:
            pid = status["pid"]
            os.kill(pid, signal.SIGTERM)

            # 等待进程结束（最多 5 秒）
            import time
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

            if self.pid_file.exists():
                self.pid_file.unlink()

            return {"success": True, "message": "已停止"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def restart(self) -> Dict[str, Any]:
        """重启刷新脚本"""
        import time

        stop_result = self.stop()
        if not stop_result["success"]:
            if "未运行" in stop_result.get("error", ""):
                return self.start()
            return stop_result

        time.sleep(1)
        return self.start()

    def get_logs(self, lines: int = 200) -> Dict[str, Any]:
        """获取日志内容"""
        if not self.log_file.exists():
            return {"success": False, "error": "日志文件不存在"}

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            return {
                "success": True,
                "lines": log_lines,
                "total_lines": len(all_lines)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# 延迟初始化单例
_refresh_cookie_service_instance = None


def get_refresh_cookie_service():
    global _refresh_cookie_service_instance
    if _refresh_cookie_service_instance is None:
        _refresh_cookie_service_instance = RefreshCookieService()
    return _refresh_cookie_service_instance


class LazyService:
    def __getattr__(self, name):
        service = get_refresh_cookie_service()
        return getattr(service, name)


refresh_cookie_service = LazyService()
