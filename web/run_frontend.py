"""
智能垃圾分类前端展示 - 一键启动脚本

用法：
    python run_frontend.py          # 从 web/ 目录运行
    或
    python web/run_frontend.py      # 从项目根目录运行

启动后访问: http://localhost:8000
"""

import os
import sys
from pathlib import Path

# 切换到 web/ 目录，确保 uvicorn 可以找到 api_server 模块
_script_dir = Path(__file__).resolve().parent
os.chdir(str(_script_dir))

import uvicorn

if __name__ == '__main__':
    print("=" * 50)
    print("  智能垃圾分类前端展示系统")
    print("=" * 50)
    print()
    print("  启动服务中...")
    print("  前端页面: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print()
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)

    uvicorn.run(
        'api_server:app',
        host='0.0.0.0',
        port=8000,
        log_level='info',
        reload=False,
    )
