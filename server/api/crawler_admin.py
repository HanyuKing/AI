"""
管理API
提供配置、Cookie管理、启动停止等功能
"""

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse

from server.services.crawler_admin_service import crawler_admin_service

router = APIRouter()


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
        "request": request
    })


@router.get("/api/admin/crawler/config")
async def get_config():
    """获取配置"""
    config = crawler_admin_service.get_config()
    return {"success": True, "data": config}


@router.post("/api/admin/crawler/config")
async def save_config(
    schedule_enabled: bool = Form(True),
    start_hour: int = Form(7),
    end_hour: int = Form(9),
    max_votes_per_day: int = Form(10)
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
async def get_status():
    """获取运行状态"""
    status = crawler_admin_service.get_status()
    return {"success": True, "data": status}


@router.post("/api/admin/crawler/start")
async def start_crawler():
    """启动"""
    result = crawler_admin_service.start_crawler()
    return result


@router.post("/api/admin/crawler/stop")
async def stop_crawler():
    """停止"""
    result = crawler_admin_service.stop_crawler()
    return result


@router.get("/api/admin/crawler/logs")
async def get_logs(lines: int = 100):
    """获取日志"""
    result = crawler_admin_service.get_logs(lines)
    return result


@router.post("/api/admin/crawler/logs/clear")
async def clear_logs():
    """清空日志"""
    result = crawler_admin_service.clear_logs()
    return result


@router.get("/api/admin/crawler/cookies")
async def get_cookies():
    """获取所有Cookie列表（带投票统计）"""
    cookies = crawler_admin_service.get_cookies()
    return {"success": True, "data": cookies}


@router.post("/api/admin/crawler/cookies")
async def add_cookie(name: str = Form(...), cookie: str = Form(...)):
    """添加Cookie"""
    result = crawler_admin_service.add_cookie(name, cookie)
    return result


@router.put("/api/admin/crawler/cookies/{index}")
async def update_cookie(index: int, name: str = Form(...), cookie: str = Form(...)):
    """更新Cookie"""
    result = crawler_admin_service.update_cookie(index, name, cookie)
    return result


@router.delete("/api/admin/crawler/cookies/{index}")
async def delete_cookie(index: int):
    """删除Cookie"""
    result = crawler_admin_service.delete_cookie(index)
    return result


@router.get("/api/admin/crawler/cookies/{cookie_id}/logs")
async def get_cookie_vote_logs(cookie_id: str, days: int = 7):
    """获取指定Cookie的投票日志"""
    result = crawler_admin_service.get_cookie_vote_logs(cookie_id, days)
    return result

