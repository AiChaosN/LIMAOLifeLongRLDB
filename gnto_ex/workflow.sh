# 1.终止原来开启的数据库服务器
sudo systemctl stop postgresql
/home/AiChaosN/Project/Phd/project/postgresql-12.5/bin/pg_ctl -D /home/AiChaosN/Project/Phd/project/databases start -l logfile
sudo systemctl status postgresql

/home/AiChaosN/Project/Phd/project/postgresql-12.5/bin/pg_ctl -D /home/AiChaosN/Project/Phd/project/databases stop

# 2开启bao服务器
cd bao_server && python3 main.py

# 3.运行实验
python3 run_queries_assorted.py --seed 11

# 单独测试bao服务器
python3 bao_server/baoctl.py --retrain

# 单独测试bao服务器扩展对于sql的优化效果
psql -h localhost -U AiChaosN -d imdbload -f /home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/tmp/tmp.sql