# 后台运行指南

## Linux/macOS 系统

### 方式1：使用 nohup（推荐，最简单）

```bash
# ⚠️ 重要：必须使用 -u 参数禁用Python输出缓冲，否则日志不会实时写入
cd scripts/zaohaowu
nohup python3 -u zaohaowu_crawler.py > crawler.log 2>&1 &

# 或者设置环境变量
PYTHONUNBUFFERED=1 nohup python3 zaohaowu_crawler.py > crawler.log 2>&1 &

# 查看进程
ps aux | grep zaohaowu_crawler

# 查看日志（实时）
tail -f crawler.log

# 停止（替换 PID 为实际进程ID）
kill <PID>
```

**为什么看不到日志？**
- Python默认会缓冲输出，需要添加 `-u` 参数或设置 `PYTHONUNBUFFERED=1`
- 确保在脚本所在目录运行，日志文件会在当前目录生成
- 如果使用相对路径，确保路径正确

### 方式2：使用 screen（推荐用于SSH连接）

```bash
# 安装 screen（如果未安装）
# macOS: brew install screen
# Ubuntu/Debian: sudo apt-get install screen

# 进入脚本目录
cd scripts/zaohaowu

# 创建新的screen会话
screen -S crawler

# 在screen中运行脚本
python3 zaohaowu_crawler.py

# 按 Ctrl+A 然后按 D 来分离会话（脚本继续运行）

# 重新连接会话
screen -r crawler

# 查看所有会话
screen -ls

# 终止会话
screen -X -S crawler quit
```

### 方式3：使用 tmux

```bash
# 安装 tmux（如果未安装）
# macOS: brew install tmux
# Ubuntu/Debian: sudo apt-get install tmux

# 进入脚本目录
cd scripts/zaohaowu

# 创建新的tmux会话
tmux new -s crawler

# 在tmux中运行脚本
python3 zaohaowu_crawler.py

# 按 Ctrl+B 然后按 D 来分离会话

# 重新连接会话
tmux attach -t crawler

# 查看所有会话
tmux ls

# 终止会话
tmux kill-session -t crawler
```

### 方式4：使用 systemd 服务（Linux系统，推荐用于服务器）

创建服务文件 `/etc/systemd/system/zaohaowu-crawler.service`：

```ini
[Unit]
Description=早好物爬虫定时任务
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/scripts/zaohaowu
ExecStart=/usr/bin/python3 -u /path/to/scripts/zaohaowu/zaohaowu_crawler.py
Restart=always
RestartSec=10
StandardOutput=append:/path/to/scripts/zaohaowu/crawler.log
StandardError=append:/path/to/scripts/zaohaowu/crawler.log

[Install]
WantedBy=multi-user.target
```

使用服务：

```bash
# 重新加载systemd配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start zaohaowu-crawler

# 设置开机自启
sudo systemctl enable zaohaowu-crawler

# 查看状态
sudo systemctl status zaohaowu-crawler

# 查看日志
sudo journalctl -u zaohaowu-crawler -f

# 停止服务
sudo systemctl stop zaohaowu-crawler

# 禁用开机自启
sudo systemctl disable zaohaowu-crawler
```

## Windows 系统

### 方式1：使用任务计划程序（推荐）

1. 打开"任务计划程序"（Task Scheduler）
2. 创建基本任务
3. 设置触发器：每天或按需
4. 设置操作：启动程序
   - 程序：`python.exe` 或 `pythonw.exe`（无窗口）
   - 参数：`-u zaohaowu_crawler.py`
   - 起始于：`C:\path\to\scripts\zaohaowu`
5. 完成创建

### 方式2：使用 PowerShell 后台作业

```powershell
# 进入脚本目录
cd C:\path\to\scripts\zaohaowu

# 启动后台作业
Start-Job -ScriptBlock { 
    Set-Location "C:\path\to\scripts\zaohaowu"
    python -u zaohaowu_crawler.py 
}

# 查看作业
Get-Job

# 查看作业输出
Receive-Job -Id <JobId>

# 停止作业
Stop-Job -Id <JobId>
Remove-Job -Id <JobId>
```

### 方式3：使用 VBScript（无窗口运行）

创建 `start_crawler.vbs`：

```vbscript
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw.exe -u zaohaowu_crawler.py", 0, False
Set WshShell = Nothing
```

双击运行即可，无窗口后台运行。

## 查看日志

### Linux/macOS

```bash
# 进入脚本目录
cd scripts/zaohaowu

# 实时查看日志（推荐）
tail -f crawler.log

# 查看最后100行
tail -n 100 crawler.log

# 搜索日志
grep "成功" crawler.log
```

### 找不到日志文件？

1. **检查当前目录**：确保在脚本所在目录运行
   ```bash
   cd scripts/zaohaowu
   ls -la crawler.log
   ```

2. **检查 nohup.out**：如果没有指定输出文件，可能输出到 `~/nohup.out`
   ```bash
   cat ~/nohup.out
   ```

### Windows

```cmd
# 进入脚本目录
cd C:\path\to\scripts\zaohaowu

# 查看日志
type crawler.log

# 实时查看（使用PowerShell）
Get-Content crawler.log -Wait -Tail 50
```

## 检查运行状态

### Linux/macOS

```bash
# 查看进程
ps aux | grep zaohaowu_crawler

# 检查进程是否运行
ps aux | grep zaohaowu_crawler | grep -v grep
```

### Windows

```cmd
# 查看Python进程
tasklist | findstr python

# 在任务管理器中查看
# Ctrl+Shift+Esc 打开任务管理器
```

## 常见问题

### 1. 脚本启动后立即退出

- 检查Python路径是否正确
- 检查依赖是否安装：`pip install httpx`
- 查看日志文件中的错误信息

### 2. 无法写入日志文件

- 检查目录权限
- 确保有写入权限：`chmod 755 scripts/zaohaowu/`

### 3. 定时任务不执行

- 检查系统时间是否正确
- 检查配置文件中的时间范围设置
- 查看日志确认是否在等待时间

### 4. 进程意外退出

- 使用 `Restart=always`（systemd）或 `screen -dmS` 自动重启
- 检查系统资源（内存、磁盘空间）
- 查看系统日志排查错误

## 推荐方案

- **个人电脑（macOS/Linux）**：使用 `screen` 或 `tmux`
- **个人电脑（Windows）**：使用任务计划程序
- **服务器（Linux）**：使用 systemd 服务
- **临时运行**：使用 `nohup` 命令

