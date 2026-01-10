"""
爬虫管理API
提供爬虫配置、Cookie管理、启动停止等功能
需要密码验证
"""

import hashlib
import os
from fastapi import APIRouter, Request, HTTPException, Depends, Form, Header
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional

from server.services.crawler_admin_service import crawler_admin_service

router = APIRouter()

# 管理密码（使用SHA256哈希存储，默认密码：admin123）
# 可以通过环境变量 ADMIN_PASSWORD_HASH 设置
ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH", 
    "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"  # admin123的SHA256
)


def verify_auth(authorization: Optional[str] = Header(None)) -> bool:
    """验证HTTP Basic认证"""
    if not authorization:
        raise HTTPException(status_code=401, detail="需要认证", headers={"WWW-Authenticate": "Basic"})
    
    try:
        # 解析Basic认证
        scheme, credentials = authorization.split(' ', 1)
        if scheme.lower() != 'basic':
            raise HTTPException(status_code=401, detail="认证方式错误")
        
        import base64
        decoded = base64.b64decode(credentials).decode('utf-8')
        username, password = decoded.split(':', 1)
        
        # 验证密码
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if username == "admin" and password_hash == ADMIN_PASSWORD_HASH:
            return True
        else:
            raise HTTPException(status_code=401, detail="用户名或密码错误", headers={"WWW-Authenticate": "Basic"})
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=401, detail="认证失败", headers={"WWW-Authenticate": "Basic"})


@router.get("/admin/crawler", response_class=HTMLResponse)
async def crawler_admin_page(request: Request, authorization: Optional[str] = Header(None)):
    """
    爬虫管理页面（需要密码验证）
    入口不显示在前端导航中
    访问地址: /admin/crawler
    默认用户名: admin
    默认密码: admin123
    """
    verify_auth(authorization)
    
    from server.core.config import settings
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))
    
    return templates.TemplateResponse("crawler_admin.html", {
        "request": request
    })


@router.get("/api/admin/crawler/config")
async def get_config(authorization: Optional[str] = Header(None)):
    """获取爬虫配置"""
    verify_auth(authorization)
    
    config = crawler_admin_service.get_config()
    return {"success": True, "data": config}


@router.post("/api/admin/crawler/config")
async def save_config(
    cookies: str = Form(...),
    schedule_enabled: bool = Form(True),
    start_hour: int = Form(7),
    end_hour: int = Form(9),
    max_votes_per_day: int = Form(10),
    authorization: Optional[str] = Header(None)
):
    """保存爬虫配置（只保存定时任务和投票限制，Cookie通过专门的接口管理）"""
    verify_auth(authorization)
    
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
async def get_status(authorization: Optional[str] = Header(None)):
    """获取爬虫运行状态"""
    verify_auth(authorization)
    
    status = crawler_admin_service.get_status()
    return {"success": True, "data": status}


@router.post("/api/admin/crawler/start")
async def start_crawler(authorization: Optional[str] = Header(None)):
    """启动爬虫"""
    verify_auth(authorization)
    
    result = crawler_admin_service.start_crawler()
    return result


@router.post("/api/admin/crawler/stop")
async def stop_crawler(authorization: Optional[str] = Header(None)):
    """停止爬虫"""
    verify_auth(authorization)
    
    result = crawler_admin_service.stop_crawler()
    return result


@router.get("/api/admin/crawler/logs")
async def get_logs(
    lines: int = 100,
    authorization: Optional[str] = Header(None)
):
    """获取日志"""
    verify_auth(authorization)
    
    result = crawler_admin_service.get_logs(lines)
    return result


@router.post("/api/admin/crawler/logs/clear")
async def clear_logs(authorization: Optional[str] = Header(None)):
    """清空日志"""
    verify_auth(authorization)
    
    result = crawler_admin_service.clear_logs()
    return result


@router.get("/api/admin/crawler/cookies")
async def get_cookies(authorization: Optional[str] = Header(None)):
    """获取所有Cookie列表（带投票统计）"""
    verify_auth(authorization)
    
    cookies = crawler_admin_service.get_cookies()
    return {"success": True, "data": cookies}


@router.post("/api/admin/crawler/cookies")
async def add_cookie(
    name: str = Form(...),
    cookie: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    """添加Cookie"""
    verify_auth(authorization)
    
    result = crawler_admin_service.add_cookie(name, cookie)
    return result


@router.put("/api/admin/crawler/cookies/{index}")
async def update_cookie(
    index: int,
    name: str = Form(...),
    cookie: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    """更新Cookie"""
    verify_auth(authorization)
    
    result = crawler_admin_service.update_cookie(index, name, cookie)
    return result


@router.delete("/api/admin/crawler/cookies/{index}")
async def delete_cookie(
    index: int,
    authorization: Optional[str] = Header(None)
):
    """删除Cookie"""
    verify_auth(authorization)
    
    result = crawler_admin_service.delete_cookie(index)
    return result


@router.get("/api/admin/crawler/cookies/{cookie_id}/logs")
async def get_cookie_vote_logs(
    cookie_id: str,
    days: int = 7,
    authorization: Optional[str] = Header(None)
):
    """获取指定Cookie的投票日志"""
    verify_auth(authorization)
    
    result = crawler_admin_service.get_cookie_vote_logs(cookie_id, days)
    return result

