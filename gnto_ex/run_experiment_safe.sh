#!/bin/bash

# 获取当前脚本所在目录的上一级目录（即项目根目录 LIMAOLifeLongRLDB）
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$PROJECT_ROOT"

# 定义时间戳和归档目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="$PROJECT_ROOT/gnto_ex/archive/$TIMESTAMP"
LOG_DIR="$ARCHIVE_DIR/logs"

echo "=== 初始化归档目录: $ARCHIVE_DIR ==="
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib:/home/AiChaosN/miniconda3/envs/limao/lib/python3.8/site-packages/nvidia/cudnn/lib
echo "已更新 LD_LIBRARY_PATH 以包含 CUDA 和 cuDNN 库路径"

# 1. 归档旧数据 (防止覆盖)
echo "正在归档旧的数据库、模型和日志..."

# 归档数据库文件 (匹配新的命名规范 history_*.db)
if [ -f "bao_server/history_BAO.db" ]; then mv "bao_server/history_BAO.db" "$ARCHIVE_DIR/"; fi
if [ -f "bao_server/history_GNTO.db" ]; then mv "bao_server/history_GNTO.db" "$ARCHIVE_DIR/"; fi
# 兼容旧的命名 (以防万一)
if [ -f "bao_server/bao.db" ]; then mv "bao_server/bao.db" "$ARCHIVE_DIR/"; fi
if [ -f "bao_server/bao_gnto.db" ]; then mv "bao_server/bao_gnto.db" "$ARCHIVE_DIR/"; fi

# 归档进度配置
if [ -f "bao_server/current_progress.cfg" ]; then mv "bao_server/current_progress.cfg" "$ARCHIVE_DIR/"; fi

# 归档模型文件夹 (bao_default_model_gnto 等)
if [ -d "bao_server/bao_default_model_gnto" ]; then mv "bao_server/bao_default_model_gnto" "$ARCHIVE_DIR/"; fi
if [ -d "bao_server/bao_previous_model_gnto" ]; then mv "bao_server/bao_previous_model_gnto" "$ARCHIVE_DIR/"; fi
# 如果有普通的 bao 模型文件夹也归档 (根据 main.py 中的常量，可能是 bao_default_model 等，这里假设名字)
# 也可以归档 nn_weights 等如果它们散落在外面

echo "归档完成。"

# 2. 数据库环境准备
echo "=== 重启 PostgreSQL ==="
sudo systemctl stop postgresql
/home/AiChaosN/Project/Phd/project/postgresql-12.5/bin/pg_ctl -D /home/AiChaosN/Project/Phd/project/databases stop 2>/dev/null
sleep 2
/home/AiChaosN/Project/Phd/project/postgresql-12.5/bin/pg_ctl -D /home/AiChaosN/Project/Phd/project/databases start -l "$LOG_DIR/pg_server.log"

# 3. 启动 Bao 服务器 (后台运行)
echo "=== 启动 Bao Server ==="
cd bao_server
# 使用 nohup 后台运行，并将日志输出到归档目录
nohup python3 main.py > "$LOG_DIR/bao_server_run.log" 2>&1 &
BAO_PID=$!
echo "Bao Server PID: $BAO_PID"
cd ..

# 等待服务器启动
echo "等待 5 秒让服务器初始化..."
sleep 5

# 4. 运行实验
echo "=== 开始运行实验 (run_queries_assorted.py) ==="
# 使用 tee 同时输出到屏幕和日志文件
python3 run_queries_assorted.py --seed 11 2>&1 | tee "$LOG_DIR/experiment_output.log"

# 5. 清理工作
echo "=== 实验结束，清理进程 ==="
kill $BAO_PID
echo "Bao Server 已停止。"

echo "所有结果已保存至: $ARCHIVE_DIR"
