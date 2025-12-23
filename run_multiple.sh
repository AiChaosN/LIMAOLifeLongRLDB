#!/bin/bash
# source /mydata/anaconda3/bin/activate balsa # Assuming conda is already active or managed by user

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
BAO_SERVER="$PROJECT_ROOT/bao_server"
DB_PATH="/home/AiChaosN/Project/Phd/project/databases"
LOG_FILE="$PROJECT_ROOT/logfile"

function restart_db() {
    pg_ctl -D "$DB_PATH" restart -l "$LOG_FILE"
}

function clean_bao() {
    rm -rf "$BAO_SERVER/bao_default_model"
    rm -rf "$BAO_SERVER/bao_previous_model"
    rm "$BAO_SERVER/bao.db"
}

############################################
cd "$PROJECT_ROOT"
rm -f "$LOG_FILE"
clean_bao
restart_db

cd "$BAO_SERVER"
python ./main.py > "$PROJECT_ROOT/arm_selection_1.log" 2>&1 &
MAIN_PID=$!
sleep 10 
python "$PROJECT_ROOT/run_queries_assorted.py" --seed 10 > "$PROJECT_ROOT/bao_assorted_1.log" 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd "$PROJECT_ROOT"
rm -f "$LOG_FILE"
clean_bao
restart_db

cd "$BAO_SERVER"
python ./main.py > "$PROJECT_ROOT/arm_selection_2.log" 2>&1 &
MAIN_PID=$!
sleep 10
python "$PROJECT_ROOT/run_queries_assorted.py" --seed 20 > "$PROJECT_ROOT/bao_assorted_2.log" 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd "$PROJECT_ROOT"
rm -f "$LOG_FILE"
clean_bao
restart_db

cd "$BAO_SERVER"
python ./main.py > "$PROJECT_ROOT/arm_selection_3.log" 2>&1 &
MAIN_PID=$!
sleep 10  
python "$PROJECT_ROOT/run_queries_assorted.py" --seed 30 > "$PROJECT_ROOT/bao_assorted_3.log" 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd "$PROJECT_ROOT"
rm -f "$LOG_FILE"
clean_bao
restart_db

cd "$BAO_SERVER"
python ./main.py > "$PROJECT_ROOT/arm_selection_4.log" 2>&1 &
MAIN_PID=$!
sleep 10
python "$PROJECT_ROOT/run_queries_assorted.py" --seed 40 > "$PROJECT_ROOT/bao_assorted_4.log" 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null
############################################
cd "$PROJECT_ROOT"
rm -f "$LOG_FILE"
clean_bao
restart_db

cd "$BAO_SERVER"
python ./main.py > "$PROJECT_ROOT/arm_selection_5.log" 2>&1 &
MAIN_PID=$!
sleep 10  
python "$PROJECT_ROOT/run_queries_assorted.py" --seed 50 > "$PROJECT_ROOT/bao_assorted_5.log" 2>&1
kill $MAIN_PID
wait $MAIN_PID 2>/dev/null

cd "$PROJECT_ROOT"
rm -f "$LOG_FILE"
clean_bao
restart_db
