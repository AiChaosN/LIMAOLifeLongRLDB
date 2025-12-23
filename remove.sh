#!/bin/bash
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
BAO_SERVER="$PROJECT_ROOT/bao_server"

rm "$BAO_SERVER/bao.db"
rm -rf "$BAO_SERVER/bao_default_model"
rm -rf "$BAO_SERVER/bao_previous_model"
rm "$BAO_SERVER/current_progress.cfg"
rm "$PROJECT_ROOT/logfile"

# Assuming databases are in a standard location relative to project or defined by env
# If databases path is truly fixed, we can keep it or use an env var.
# For now, I'll assume user wants to replace /mydata with something local or dynamic if possible.
# But unlike python scripts, shell scripts are often run from specific places.
# I'll use a variable for the DB path if I can find where it should be.
# Based on grep earlier: /home/AiChaosN/Project/Phd/project/databases seems to be the one.

DB_PATH="/home/AiChaosN/Project/Phd/project/databases"
pg_ctl -D "$DB_PATH" restart -l "$PROJECT_ROOT/logfile"