"""
管理API
提供配置、Cookie管理、启动停止等功能
"""

import os
import hashlib
from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from server.services.crawler_admin_service import crawler_admin_service

router = APIRouter()

# 默认密码：admin123（可通过环境变量 ADMIN_PASSWORD 设置）
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()


def check_login(request: Request) -> bool:
    """检查是否已登录"""
    return request.session.get("admin_logged_in", False)


def require_login(request: Request):
    """要求登录的依赖"""
    if not check_login(request):
        raise HTTPException(status_code=401, detail="需要登录")


@router.post("/api/admin/crawler/login")
async def login(request: Request, password: str = Form(...)):
    """登录"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash == ADMIN_PASSWORD_HASH:
        request.session["admin_logged_in"] = True
        return {"success": True, "message": "登录成功"}
    else:
        return {"success": False, "error": "密码错误"}


@router.post("/api/admin/crawler/logout")
async def logout(request: Request):
    """登出"""
    request.session.pop("admin_logged_in", None)
    return {"success": True, "message": "已登出"}


@router.get("/api/admin/crawler/check-login")
async def check_login_status(request: Request):
    """检查登录状态"""
    return {"logged_in": check_login(request)}


@router.get("/admin/crawler", response_class=HTMLResponse)
async def crawler_admin_page(request: Request):
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
        "logged_in": check_login(request)
    })


@router.get("/api/admin/crawler/config")
async def get_config(request: Request):
    """获取配置"""
    require_login(request)
    config = crawler_admin_service.get_config()
    return {"success": True, "data": config}


@router.post("/api/admin/crawler/config")
async def save_config(
    request: Request,
    schedule_enabled: bool = Form(True),
    start_hour: int = Form(7),
    end_hour: int = Form(9),
    max_votes_per_day: int = Form(10)
):
    """保存配置（只保存定时任务和投票限制，Cookie通过专门的接口管理）"""
    require_login(request)
    
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
async def get_status(request: Request):
    """获取运行状态"""
    require_login(request)
    status = crawler_admin_service.get_status()
    return {"success": True, "data": status}


@router.post("/api/admin/crawler/start")
async def start_crawler(request: Request):
    """启动"""
    require_login(request)
    result = crawler_admin_service.start_crawler()
    return result


@router.post("/api/admin/crawler/stop")
async def stop_crawler(request: Request):
    """停止"""
    require_login(request)
    result = crawler_admin_service.stop_crawler()
    return result


@router.post("/api/admin/crawler/restart")
async def restart_crawler(request: Request):
    """重启"""
    require_login(request)
    result = crawler_admin_service.restart_crawler()
    return result


@router.get("/api/admin/crawler/logs")
async def get_logs(request: Request, lines: int = 100):
    """获取日志"""
    require_login(request)
    result = crawler_admin_service.get_logs(lines)
    return result


@router.post("/api/admin/crawler/logs/clear")
async def clear_logs(request: Request):
    """清空日志"""
    require_login(request)
    result = crawler_admin_service.clear_logs()
    return result


@router.get("/api/admin/crawler/cookies")
async def get_cookies(request: Request):
    """获取所有Cookie列表（带投票统计）"""
    require_login(request)
    cookies = crawler_admin_service.get_cookies()
    return {"success": True, "data": cookies}


@router.post("/api/admin/crawler/cookies")
async def add_cookie(request: Request, name: str = Form(...), cookie: str = Form(...)):
    """添加Cookie"""
    require_login(request)
    result = crawler_admin_service.add_cookie(name, cookie)
    return result


@router.put("/api/admin/crawler/cookies/{index}")
async def update_cookie(request: Request, index: int, name: str = Form(...), cookie: str = Form(...)):
    """更新Cookie"""
    require_login(request)
    result = crawler_admin_service.update_cookie(index, name, cookie)
    return result


@router.delete("/api/admin/crawler/cookies/{index}")
async def delete_cookie(request: Request, index: int):
    """删除Cookie"""
    require_login(request)
    result = crawler_admin_service.delete_cookie(index)
    return result


@router.get("/api/admin/crawler/cookies/{cookie_id}/logs")
async def get_cookie_vote_logs(request: Request, cookie_id: str, days: int = 7):
    """获取指定Cookie的投票日志"""
    require_login(request)
    result = crawler_admin_service.get_cookie_vote_logs(cookie_id, days)
    return result

