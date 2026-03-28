#!/bin/bash
# Training Monitor Script
# Usage: ./scripts/monitor_training.sh

clear
echo "======================================================================"
echo "🔍 SESSION-BASED MODEL TRAINING MONITOR"
echo "======================================================================"
echo ""

# Check if training process exists
TRAIN_PID=$(ps aux | grep train_session_based | grep python | grep -v grep | awk '{print $2}' | tail -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ NO TRAINING PROCESS FOUND"
    echo ""
    echo "Training may have:"
    echo "  - Completed successfully"
    echo "  - Crashed with an error"
    echo "  - Not started yet"
    echo ""
    echo "Check the log file:"
    echo "  tail -50 training_session.log"
    echo ""
    exit 1
fi

# Get process info
PROC_INFO=$(ps aux | grep $TRAIN_PID | grep -v grep)
CPU=$(echo $PROC_INFO | awk '{print $3}')
MEM=$(echo $PROC_INFO | awk '{print $4}')
RUNTIME=$(echo $PROC_INFO | awk '{print $10}')

echo "✅ TRAINING PROCESS ACTIVE"
echo "   PID: $TRAIN_PID"
echo "   CPU: ${CPU}%"
echo "   Memory: ${MEM}%"
echo "   Runtime: $RUNTIME"
echo ""

# GPU Status
echo "🎮 GPU STATUS"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
while IFS=', ' read -r util mem_used mem_total temp; do
    echo "   Utilization: $util"
    echo "   Memory: $mem_used / $mem_total"
    echo "   Temperature: $temp"
done
echo ""

# Log file status
if [ -f "training_session.log" ]; then
    LOG_SIZE=$(wc -c < training_session.log)
    LOG_LINES=$(wc -l < training_session.log)
    echo "📝 LOG FILE STATUS"
    echo "   Size: $LOG_SIZE bytes"
    echo "   Lines: $LOG_LINES"
    
    if [ $LOG_SIZE -gt 1000 ]; then
        echo "   Last 5 lines:"
        tail -5 training_session.log | sed 's/^/   │ /'
    else
        echo "   Status: Buffering (normal)"
    fi
    echo ""
fi

# Estimate progress
RUNTIME_SECS=$(echo $RUNTIME | awk -F: '{if (NF==2) print $1*60+$2; else print $1*3600+$2*60+$3}')
TOTAL_ESTIMATED=1200  # 20 minutes
REMAINING=$((TOTAL_ESTIMATED - RUNTIME_SECS))
REMAINING_MIN=$((REMAINING / 60))

if [ $REMAINING_MIN -gt 0 ]; then
    echo "⏱️  ESTIMATED TIME REMAINING: ~${REMAINING_MIN} minutes"
else
    echo "⏱️  ESTIMATED TIME REMAINING: Finishing soon..."
fi
echo ""

# Progress bar
PROGRESS=$((RUNTIME_SECS * 100 / TOTAL_ESTIMATED))
if [ $PROGRESS -gt 100 ]; then
    PROGRESS=100
fi

BAR_LENGTH=50
FILLED=$((PROGRESS * BAR_LENGTH / 100))
EMPTY=$((BAR_LENGTH - FILLED))

printf "📊 PROGRESS: ["
printf "%${FILLED}s" | tr ' ' '█'
printf "%${EMPTY}s" | tr ' ' '░'
printf "] ${PROGRESS}%%\n"
echo ""

echo "======================================================================"
echo "💡 MONITORING COMMANDS:"
echo "   Watch this status: watch -n 5 ./scripts/monitor_training.sh"
echo "   Watch GPU only:    watch -n 2 nvidia-smi"
echo "   Follow log:        tail -f training_session.log"
echo "   Kill training:     kill $TRAIN_PID"
echo "======================================================================"
