# 爬虫管理页面使用说明

## 访问地址

管理页面地址：`/admin/crawler`

**注意**：此页面不在前端导航中显示，需要直接访问URL。

## 默认登录信息

- **用户名**：`admin`
- **密码**：`admin123`

## 修改密码

可以通过环境变量修改密码哈希值：

```bash
# 生成新密码的SHA256哈希
python3 -c "import hashlib; print(hashlib.sha256('your_password'.encode()).hexdigest())"

# 设置环境变量
export ADMIN_PASSWORD_HASH="生成的哈希值"
```

或者在代码中修改 `server/api/crawler_admin.py` 中的 `ADMIN_PASSWORD_HASH` 变量。

## 功能说明

### 1. 运行状态
- 显示爬虫当前运行状态（运行中/已停止）
- 显示进程ID（PID）
- 显示今日总投票数
- 提供启动/停止/刷新按钮

### 2. 配置管理
- **Cookies管理**：每行一个Cookie，支持多个Cookie并发执行
- **定时任务配置**：
  - 启用/禁用定时任务
  - 设置运行时间范围（开始时间-结束时间，24小时制）
  - 设置每日投票限制
- 保存配置后会自动更新到 `scripts/zaohaowu/cookies.json` 文件

### 3. 日志查看
- 实时查看爬虫运行日志（最后100行）
- 支持刷新日志
- 支持清空日志

## 技术实现

- **后端**：FastAPI + Python
- **前端**：HTML + JavaScript + Tailwind CSS
- **认证**：HTTP Basic Authentication（用户名密码）
- **服务**：`server/services/crawler_admin_service.py` 处理所有业务逻辑
- **API**：`server/api/crawler_admin.py` 提供RESTful API接口

## 注意事项

1. 管理页面需要密码验证，请妥善保管密码
2. 修改配置后需要保存才能生效
3. 启动/停止爬虫操作会立即生效
4. 日志文件位置：`scripts/zaohaowu/crawler.log`
5. PID文件位置：`scripts/zaohaowu/data/crawler.pid`

