#!/bin/bash

# Hyperparameter Search Script for Online Learning (online_learning.py)
# Sweeps: finetune_label_ratio, intermediate_dim, update ratio, solver weights

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate DDPM

# Verify environment
echo "Active environment: $(conda info --envs | grep '\*' | awk '{print $1}')"
python --version

# Check GPU memory before starting
GPU_ID=1
echo "Checking GPU ${GPU_ID} memory..."
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -i ${GPU_ID}

# Create output directories
mkdir -p logs_online_search

# Summary CSV  (one row per scenario per experiment, written incrementally)
SUMMARY_CSV="logs_online_search/summary_$(date +%Y%m%d_%H%M%S).csv"
echo "exp_name,label_ratio,hidden_dim,pinn_batches,encoder_batches,solver_weights,"\
"w_p_balance,w_q_balance,w_theta_balance,w_active,w_reactive,w_voltage,w_line,"\
"exit_code,scenario_type,scenario_name,"\
"pred_p,pred_q,pred_cost,pred_active,pred_reactive,pred_voltage,pred_line,pred_pf_success_rate,"\
"baseline_p,baseline_q,baseline_cost,baseline_active,baseline_reactive,baseline_voltage,baseline_line,"\
"log_file" \
    > "$SUMMARY_CSV"
echo "Summary CSV: $SUMMARY_CSV"

# Helper: read the per-experiment results CSV and append all its rows to SUMMARY_CSV.
# Only called when exit_code == 0.  Falls back to NA row if the file is missing.
# Args: results_csv_path exp_name label_ratio hidden_dim pinn_batches encoder_batches
#       solver_weights w_p w_q w_theta w_active w_reactive w_voltage w_line
#       exit_code log_file
append_scenario_rows() {
    python3 - "$SUMMARY_CSV" "$@" <<'PYEOF'
import sys, csv, os

summary_path = sys.argv[1]
results_path = sys.argv[2]
(
    exp_name, label_ratio, hidden_dim, pinn_batches, encoder_batches, solver_weights,
    w_p, w_q, w_theta, w_active, w_reactive, w_voltage, w_line,
    exit_code, log_file
) = sys.argv[3:]

prefix = [
    exp_name, label_ratio, hidden_dim, pinn_batches, encoder_batches, solver_weights,
    w_p, w_q, w_theta, w_active, w_reactive, w_voltage, w_line,
    exit_code
]

metric_cols = [
    'scenario_type', 'scenario_name',
    'pred_p', 'pred_q', 'pred_cost', 'pred_active', 'pred_reactive', 'pred_voltage', 'pred_line',
    'pred_pf_success_rate',
    'baseline_p', 'baseline_q', 'baseline_cost', 'baseline_active',
    'baseline_reactive', 'baseline_voltage', 'baseline_line',
]

try:
    with open(results_path) as f:
        rows = list(csv.DictReader(f))
except Exception:
    rows = []

with open(summary_path, 'a', newline='') as out:
    writer = csv.writer(out)
    if rows:
        for row in rows:
            writer.writerow(prefix + [row.get(c, 'NA') for c in metric_cols] + [log_file])
    else:
        writer.writerow(prefix + ['NA'] * len(metric_cols) + [log_file])
PYEOF
}

# Failed-experiment log
FAILED_LOG="logs_online_search/failed_experiments_$(date +%Y%m%d_%H%M%S).txt"
echo "Failed Experiments Log - Created at $(date)" > "$FAILED_LOG"
echo "==========================================" >> "$FAILED_LOG"

# ─── Sweep dimensions ───────────────────────────────────────────────────────

# Fraction of 2000 fine-tune samples that carry labels (supervised step)
LABEL_RATIOS=(
    "0.0"
    "0.1"
    "0.2"
)

# Encoder intermediate dimension
HIDDEN_DIMS=(
    64
    128
)

# step_pinn_batches : step_encoder_batches
UPDATE_RATIOS=(
    "1:1"
    "2:1"
    "4:1"
    "8:1"
)

# PINN-branch weight : Encoder-branch weight
# "1:1"  → (w_p,w_q,w_theta)=1   (w_active,w_reactive,w_voltage,w_line)=1
# "10:1" → (w_p,w_q,w_theta)=10  encoder weights=1
# "1:10" → (w_p,w_q,w_theta)=1   encoder weights=10
SOLVER_WEIGHTS=(
    "1:1"
    "10:1"
    "1:10"
)

# ─── Fixed parameters ───────────────────────────────────────────────────────
PRETRAIN_EPOCHS=1000
FINETUNE_EPOCHS=1000
BATCH_SIZE=128
LR=5e-5
W_COST=1e-9

# ─── Helper functions ────────────────────────────────────────────────────────

parse_ratio() {
    local ratio=$1
    local pinn_batches=$(echo $ratio | cut -d':' -f1)
    local encoder_batches=$(echo $ratio | cut -d':' -f2)
    echo "$pinn_batches $encoder_batches"
}

parse_solver_weights() {
    local weights=$1
    local pinn_w=$(echo $weights | cut -d':' -f1)
    local enc_w=$(echo $weights | cut -d':' -f2)
    echo "$pinn_w $enc_w"
}

# ─── Count total experiments ─────────────────────────────────────────────────
TOTAL_EXPERIMENTS=0
for lr_ratio in "${LABEL_RATIOS[@]}"; do
    for hidden_dim in "${HIDDEN_DIMS[@]}"; do
        for ratio in "${UPDATE_RATIOS[@]}"; do
            for solver_w in "${SOLVER_WEIGHTS[@]}"; do
                TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
            done
        done
    done
done

echo "=========================================="
echo "Online Learning Hyperparameter Search"
echo "=========================================="
echo "Total experiments : $TOTAL_EXPERIMENTS"
echo "Label ratios      : ${LABEL_RATIOS[*]}"
echo "Hidden dims       : ${HIDDEN_DIMS[*]}"
echo "Update ratios     : ${UPDATE_RATIOS[*]}"
echo "Solver weights    : ${SOLVER_WEIGHTS[*]}"
echo "Pretrain epochs   : $PRETRAIN_EPOCHS"
echo "Finetune epochs   : $FINETUNE_EPOCHS"
echo "Batch size        : $BATCH_SIZE"
echo "Learning rate     : $LR"
echo "=========================================="
echo ""

CURRENT_EXP=0

# ─── Main sweep ──────────────────────────────────────────────────────────────
for LABEL_RATIO in "${LABEL_RATIOS[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
        for ratio in "${UPDATE_RATIOS[@]}"; do
            read PINN_BATCHES ENCODER_BATCHES <<< $(parse_ratio $ratio)

            for solver_w in "${SOLVER_WEIGHTS[@]}"; do
                read PINN_W ENC_W <<< $(parse_solver_weights $solver_w)

                # Map solver weight tag → actual float values
                if [ "$PINN_W" == "1" ] && [ "$ENC_W" == "1" ]; then
                    W_P_BALANCE=1.0;  W_Q_BALANCE=1.0;  W_THETA_BALANCE=1.0
                    W_ACTIVE=1.0;     W_REACTIVE=1.0;   W_VOLTAGE=1.0;  W_LINE=1.0
                elif [ "$PINN_W" == "10" ] && [ "$ENC_W" == "1" ]; then
                    W_P_BALANCE=10.0; W_Q_BALANCE=10.0; W_THETA_BALANCE=10.0
                    W_ACTIVE=1.0;     W_REACTIVE=1.0;   W_VOLTAGE=1.0;  W_LINE=1.0
                elif [ "$PINN_W" == "1" ] && [ "$ENC_W" == "10" ]; then
                    W_P_BALANCE=1.0;  W_Q_BALANCE=1.0;  W_THETA_BALANCE=1.0
                    W_ACTIVE=10.0;    W_REACTIVE=10.0;  W_VOLTAGE=10.0; W_LINE=10.0
                fi

                CURRENT_EXP=$((CURRENT_EXP + 1))

                LABEL_TAG=$(echo $LABEL_RATIO | sed 's/\./_/')
                EXP_NAME="lr${LABEL_TAG}_hd${HIDDEN_DIM}_ratio${ratio}_solver${solver_w}"
                LOG_FILE="logs_online_search/${EXP_NAME}.log"
                RESULTS_FILE="results_online/${EXP_NAME}_results.csv"

                echo "=========================================="
                echo "Experiment $CURRENT_EXP / $TOTAL_EXPERIMENTS: $EXP_NAME"
                echo "=========================================="
                echo "  Label ratio    : $LABEL_RATIO  (labeled=$(python3 -c "print(int($LABEL_RATIO*2000))") / 2000)"
                echo "  Hidden dim     : $HIDDEN_DIM"
                echo "  Update ratio   : ${PINN_BATCHES}:${ENCODER_BATCHES}"
                echo "  PINN weights   : ($W_P_BALANCE, $W_Q_BALANCE, $W_THETA_BALANCE)"
                echo "  Encoder weights: ($W_ACTIVE, $W_REACTIVE, $W_VOLTAGE, $W_LINE)"
                echo "  Log file       : $LOG_FILE"
                echo ""

                # Run online_learning.py
                # stdout (logging) → tee → log file + console
                # stderr (tqdm)    → console only
                python online_learning.py \
                    --gpu_id              $GPU_ID \
                    --pretrain_epochs     $PRETRAIN_EPOCHS \
                    --finetune_epochs     $FINETUNE_EPOCHS \
                    --batch_size          $BATCH_SIZE \
                    --lr                  $LR \
                    --intermediate_dim    $HIDDEN_DIM \
                    --step_pinn_batches   $PINN_BATCHES \
                    --step_encoder_batches $ENCODER_BATCHES \
                    --finetune_label_ratio $LABEL_RATIO \
                    --w_p_balance         $W_P_BALANCE \
                    --w_q_balance         $W_Q_BALANCE \
                    --w_theta_balance     $W_THETA_BALANCE \
                    --w_cost              $W_COST \
                    --w_active            $W_ACTIVE \
                    --w_reactive          $W_REACTIVE \
                    --w_voltage           $W_VOLTAGE \
                    --w_line              $W_LINE \
                    --validate_every      100 \
                    --sample_num_for_validate 20 \
                    --results_file        "$RESULTS_FILE" \
                    2> /dev/tty | tee "$LOG_FILE"

                PYTHON_EXIT_CODE=${PIPESTATUS[0]}
                if [ $PYTHON_EXIT_CODE -eq 0 ]; then
                    echo "✓ Experiment $EXP_NAME completed successfully"
                else
                    echo "✗ Experiment $EXP_NAME failed (exit code: $PYTHON_EXIT_CODE)"
                    {
                        echo "$(date '+%Y-%m-%d %H:%M:%S') - $EXP_NAME"
                        echo "  Exit code       : $PYTHON_EXIT_CODE"
                        echo "  Log file        : $LOG_FILE"
                        echo "  label_ratio     : $LABEL_RATIO"
                        echo "  hidden_dim      : $HIDDEN_DIM"
                        echo "  update_ratio    : ${PINN_BATCHES}:${ENCODER_BATCHES}"
                        echo "  solver_weights  : $solver_w"
                        echo "  w_p_balance     : $W_P_BALANCE"
                        echo "  w_q_balance     : $W_Q_BALANCE"
                        echo "  w_theta_balance : $W_THETA_BALANCE"
                        echo "  w_active        : $W_ACTIVE"
                        echo "  w_reactive      : $W_REACTIVE"
                        echo "  w_voltage       : $W_VOLTAGE"
                        echo "  w_line          : $W_LINE"
                        echo "----------------------------------------"
                    } >> "$FAILED_LOG"
                fi

                # Append one row per scenario to the summary CSV
                # Pass the experiment-specific results file; NA rows written if missing/failed
                append_scenario_rows \
                    "$RESULTS_FILE" \
                    "$EXP_NAME" "$LABEL_RATIO" "$HIDDEN_DIM" \
                    "$PINN_BATCHES" "$ENCODER_BATCHES" "$solver_w" \
                    "$W_P_BALANCE" "$W_Q_BALANCE" "$W_THETA_BALANCE" \
                    "$W_ACTIVE" "$W_REACTIVE" "$W_VOLTAGE" "$W_LINE" \
                    "$PYTHON_EXIT_CODE" "$LOG_FILE"

                echo ""

                # Brief pause to let GPU memory fully clear
                sleep 30
            done
        done
    done
done

# ─── Final report ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "Online Hyperparameter Search Completed!"
echo "=========================================="

if [ -s "$FAILED_LOG" ]; then
    FAILED_COUNT=$(grep -c "^[0-9]" "$FAILED_LOG")
    echo ""
    echo "WARNING: $FAILED_COUNT experiment(s) failed!"
    echo "Failed experiments log: $FAILED_LOG"
else
    echo ""
    echo "All $TOTAL_EXPERIMENTS experiments completed successfully!"
    rm -f "$FAILED_LOG"
fi

echo "Total experiments : $TOTAL_EXPERIMENTS"
echo "Logs saved in     : ./logs_online_search/"
echo "Results saved in  : ./results_online/"
echo "Models saved in   : ./save_model_online/"
echo "Summary CSV       : $SUMMARY_CSV"
echo "=========================================="
