#!/bin/bash
# One machine per world model: train the model if it has no checkpoint yet, then run its E1 sweep (each SCP variant
# at each margin_eps in its own process, MPPI and CEM together in one) and, with E2=1, its E2 sweep, all in parallel.
# Usage: [SEEDS=0:30] [E2=1] bash src/scripts/launch_model.sh <model config name> [margin_eps ...]
#   e.g. E2=1 bash src/scripts/launch_model.sh pg1_c_1_5_model
#        SEEDS=0:20 bash src/scripts/launch_model.sh pg1_c_1_5_model_h30 0.1 0.2 0.3
cd "$(dirname "$0")/../.." || exit 1
MODEL=$1; shift
EPS=${*:-0.1 0.2 0.3 0.5 0.0}
SEEDS=${SEEDS:-0:30}
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs

# Train only a model without a checkpoint, so an existing one (such as M2) is never retrained
DATA_DIR=$(python -c "import yaml; print(yaml.safe_load(open('configs/wmTraining/$MODEL.yaml'))['data_dir'])") || exit 1
if [ ! -d "src/save_data/$DATA_DIR/Model/state" ]; then
    echo "training $MODEL"
    TRAIN="import sys; sys.path.insert(0, 'src/scripts'); from train_world_model import main"
    python -c "$TRAIN; main('configs/wmTraining/$MODEL.yaml')" > "logs/${MODEL}_train.log" 2>&1 \
        || { echo "training failed, see logs/${MODEL}_train.log"; exit 1; }
fi

echo "running $MODEL: margin_eps $EPS, seeds $SEEDS"
for e in $EPS; do
    for p in SCP+MPPI SCP; do
        python src/scripts/run_e1.py "$MODEL" "$e" "$p" "$SEEDS" > "logs/${MODEL}_${p}_$e.log" 2>&1 &
    done
done
python src/scripts/run_e1.py "$MODEL" MPPI CEM "$SEEDS" $EPS > "logs/${MODEL}_samplers.log" 2>&1 &
[ -n "$E2" ] && python src/scripts/run_e2.py "$MODEL" > "logs/${MODEL}_e2.log" 2>&1 &
wait
echo "finished $MODEL"
