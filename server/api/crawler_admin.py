"""
管理API
提供配置、Cookie管理、启动停止等功能
"""

import os
import hashlib
import time
from fastapi import APIRouter, Request, Form, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Optional

from server.services.crawler_admin_service import crawler_admin_service
from server.services.refresh_cookie_service import refresh_cookie_service

router = APIRouter()

# 默认密码：admin123（可通过环境变量 ADMIN_PASSWORD 设置）
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

# 简单的内存存储登录token（生产环境建议使用Redis等）
_logged_in_tokens = {}  # token -> 过期时间


def check_login(token: Optional[str] = Header(None, alias="X-Auth-Token")) -> bool:
    """检查是否已登录（通过token）"""
    if not token:
        return False
    if token in _logged_in_tokens:
        # 检查是否过期（24小时）
        if time.time() < _logged_in_tokens[token]:
            return True
        else:
            # 已过期，删除
            del _logged_in_tokens[token]
    return False


def require_login(token: Optional[str] = Header(None, alias="X-Auth-Token")):
    """要求登录的依赖"""
    if not check_login(token):
        raise HTTPException(status_code=401, detail="需要登录")


@router.post("/api/admin/crawler/login")
async def login(password: str = Form(...)):
    """登录"""
    import secrets
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash == ADMIN_PASSWORD_HASH:
        # 生成token，24小时有效期
        token = secrets.token_urlsafe(32)
        _logged_in_tokens[token] = time.time() + 86400  # 24小时
        return {"success": True, "message": "登录成功", "token": token}
    else:
        return {"success": False, "error": "密码错误"}


@router.post("/api/admin/crawler/logout")
async def logout(token: Optional[str] = Header(None, alias="X-Auth-Token")):
    """登出"""
    if token and token in _logged_in_tokens:
        del _logged_in_tokens[token]
    return {"success": True, "message": "已登出"}


@router.get("/api/admin/crawler/check-login")
async def check_login_status(token: Optional[str] = Header(None, alias="X-Auth-Token")):
    """检查登录状态"""
    return {"logged_in": check_login(token)}


@router.get("/admin/crawler", response_class=HTMLResponse)
async def crawler_admin_page(request: Request, token: Optional[str] = Header(None, alias="X-Auth-Token")):
    """
    管理页面
    入口不显示在前端导航中
    访问地址: /admin/crawler
    """
    from server.core.config import settings
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
    
    return templates.TemplateResponse("crawler_admin.html", {
        "request": request,
        "logged_in": check_login(token)
    })


@router.get("/api/admin/crawler/config")
async def get_config(token: str = Depends(require_login)):
    """获取配置"""
    config = crawler_admin_service.get_config()
    return {"success": True, "data": config}


@router.post("/api/admin/crawler/config")
async def save_config(
    schedule_enabled: bool = Form(True),
    start_hour: int = Form(7),
    end_hour: int = Form(9),
    max_votes_per_day: int = Form(10),
    token: str = Depends(require_login)
):
    """保存配置（只保存定时任务和投票限制，Cookie通过专门的接口管理）"""
    
    try:
        # 获取当前Cookie列表（保持现有Cookie不变）
        current_config = crawler_admin_service.get_config()
        current_cookies = current_config.get("cookies", [])
        
        config = {
            "cookies": current_cookies,  # 保持现有Cookie列表
            "schedule": {
                "enabled": schedule_enabled,
                "start_hour": start_hour,
                "end_hour": end_hour
            },
            "max_votes_per_day": max_votes_per_day
        }
        
        result = crawler_admin_service.save_config(config)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/api/admin/crawler/status")
async def get_status(token: str = Depends(require_login)):
    """获取运行状态"""
    status = crawler_admin_service.get_status()
    return {"success": True, "data": status}


@router.post("/api/admin/crawler/start")
async def start_crawler(token: str = Depends(require_login)):
    """启动"""
    result = crawler_admin_service.start_crawler()
    return result


@router.post("/api/admin/crawler/stop")
async def stop_crawler(token: str = Depends(require_login)):
    """停止"""
    result = crawler_admin_service.stop_crawler()
    return result


@router.post("/api/admin/crawler/restart")
async def restart_crawler(token: str = Depends(require_login)):
    """重启"""
    result = crawler_admin_service.restart_crawler()
    return result


@router.get("/api/admin/refresh/status")
async def get_refresh_status(token: str = Depends(require_login)):
    """获取 Cookie 刷新脚本状态"""
    status = refresh_cookie_service.get_status()
    return {"success": True, "data": status}


@router.post("/api/admin/refresh/start")
async def start_refresh_script(token: str = Depends(require_login)):
    """启动 Cookie 刷新脚本"""
    result = refresh_cookie_service.start()
    return result


@router.post("/api/admin/refresh/stop")
async def stop_refresh_script(token: str = Depends(require_login)):
    """停止 Cookie 刷新脚本"""
    result = refresh_cookie_service.stop()
    return result


@router.post("/api/admin/refresh/restart")
async def restart_refresh_script(token: str = Depends(require_login)):
    """重启 Cookie 刷新脚本"""
    result = refresh_cookie_service.restart()
    return result


@router.get("/api/admin/refresh/logs")
async def get_refresh_logs(lines: int = 200, token: str = Depends(require_login)):
    """获取 Cookie 刷新脚本日志"""
    result = refresh_cookie_service.get_logs(lines)
    return result


@router.get("/api/admin/crawler/logs")
async def get_logs(lines: int = 100, token: str = Depends(require_login)):
    """获取日志"""
    result = crawler_admin_service.get_logs(lines)
    return result


@router.post("/api/admin/crawler/logs/clear")
async def clear_logs(token: str = Depends(require_login)):
    """清空日志"""
    result = crawler_admin_service.clear_logs()
    return result


@router.get("/api/admin/crawler/cookies")
async def get_cookies(token: str = Depends(require_login)):
    """获取所有Cookie列表（带投票统计）"""
    cookies = crawler_admin_service.get_cookies()
    return {"success": True, "data": cookies}


@router.post("/api/admin/crawler/cookies")
async def add_cookie(name: str = Form(...), cookie: str = Form(...), token: str = Depends(require_login)):
    """添加Cookie"""
    result = crawler_admin_service.add_cookie(name, cookie)
    return result


@router.put("/api/admin/crawler/cookies/{index}")
async def update_cookie(index: int, name: str = Form(...), cookie: str = Form(...), token: str = Depends(require_login)):
    """更新Cookie"""
    result = crawler_admin_service.update_cookie(index, name, cookie)
    return result


@router.delete("/api/admin/crawler/cookies/{index}")
async def delete_cookie(index: int, token: str = Depends(require_login)):
    """删除Cookie"""
    result = crawler_admin_service.delete_cookie(index)
    return result


@router.get("/api/admin/crawler/cookies/{cookie_id}/logs")
async def get_cookie_vote_logs(cookie_id: str, days: int = 7, token: str = Depends(require_login)):
    """获取指定Cookie的投票日志"""
    result = crawler_admin_service.get_cookie_vote_logs(cookie_id, days)
    return result
