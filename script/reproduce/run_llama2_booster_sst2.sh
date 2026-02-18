#!/usr/bin/env bash
set -euo pipefail

trap 'rc=$?; echo "[error] command failed at line ${LINENO} (exit=${rc})"; exit ${rc}' ERR

DEBUG="${DEBUG:-0}"
if [[ "${DEBUG}" == "1" ]]; then
  set -x
fi

export PYTHONUNBUFFERED=1

# One-click reproduction for Booster on Llama2-7B (SST2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf}"
POISON_RATIO="${POISON_RATIO:-0.1}"
ALIGN_EPOCHS="${ALIGN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
ALIGN_SAMPLE_NUM="${ALIGN_SAMPLE_NUM:-5000}"
FINETUNE_SAMPLE_NUM="${FINETUNE_SAMPLE_NUM:-1000}"
BAD_SAMPLE_NUM="${BAD_SAMPLE_NUM:-200}"
LAMB="${LAMB:-5}"
ALPHA="${ALPHA:-0.1}"
ALIGN_LR="${ALIGN_LR:-1e-3}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-4}"
CACHE_DIR="${CACHE_DIR:-cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REQUIRE_H20="${REQUIRE_H20:-1}"
ALLOW_NON_LLAMA2="${ALLOW_NON_LLAMA2:-0}"
ENABLE_NOHUP="${ENABLE_NOHUP:-1}"

MODEL_SHORT="$(basename "${MODEL_PATH}")"
ALIGN_CKPT="ckpt/${MODEL_SHORT}_smooth_${LAMB}_${ALPHA}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"
FT_CKPT="ckpt/sst2/${MODEL_SHORT}_smooth_f_${LAMB}_${ALPHA}_${POISON_RATIO}_${FINETUNE_SAMPLE_NUM}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"

POISON_OUT="data/poison/sst2/${MODEL_SHORT}_smooth_f_${LAMB}_${ALPHA}_${POISON_RATIO}_${FINETUNE_SAMPLE_NUM}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"
SST2_OUT="data/sst2/${MODEL_SHORT}_smooth_f_${LAMB}_${ALPHA}_${POISON_RATIO}_${FINETUNE_SAMPLE_NUM}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"

if [[ "${ENABLE_NOHUP}" == "1" && -z "${BOOSTER_NOHUP_CHILD:-}" ]]; then
  cd "${REPO_ROOT}"
  mkdir -p logs/reproduce
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="logs/reproduce/llama2_booster_sst2_${RUN_ID}.log"
  export BOOSTER_NOHUP_CHILD=1
  nohup bash "${BASH_SOURCE[0]}" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  PID=$!
  echo "[info] started in background (nohup)"
  echo "[info] pid: ${PID}"
  echo "[info] log: ${LOG_FILE}"
  echo "[info] monitor: tail -f ${LOG_FILE}"
  echo "[info] stop: kill ${PID}"
  exit 0
fi

echo "[info] repo root: ${REPO_ROOT}"
echo "[info] model: ${MODEL_PATH}"
echo "[info] poison ratio: ${POISON_RATIO}"
echo "[info] align lr: ${ALIGN_LR}"
echo "[info] finetune lr: ${FINETUNE_LR}"
echo "[info] lora r/alpha: ${LORA_R}/${LORA_ALPHA}"
echo "[info] require h20 check: ${REQUIRE_H20}"
echo "[info] debug mode: ${DEBUG}"

cd "${REPO_ROOT}"

mkdir -p data ckpt ckpt/sst2 "${CACHE_DIR}" data/poison data/poison/sst2 data/sst2

if [[ "${ALLOW_NON_LLAMA2}" != "1" ]] && [[ "${MODEL_PATH}" != *"Llama-2-7b-hf"* ]]; then
  echo "[error] this reproduction script is configured for Llama2-7B only."
  echo "Set MODEL_PATH=meta-llama/Llama-2-7b-hf, or ALLOW_NON_LLAMA2=1 if you really want to bypass."
  exit 1
fi

if [[ "${REQUIRE_H20}" == "1" ]]; then
  echo "[check] verifying GPU model (H20 required)..."
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[error] nvidia-smi not found, cannot verify H20 GPU."
    exit 1
  fi

  if command -v timeout >/dev/null 2>&1; then
    GPU_NAME="$(timeout 20s nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r' || true)"
  else
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r')"
  fi

  if [[ -z "${GPU_NAME}" ]]; then
    echo "[error] failed to query GPU name (nvidia-smi timeout or empty output)."
    echo "Set REQUIRE_H20=0 to bypass this guard if needed."
    exit 1
  fi

  echo "[info] detected gpu: ${GPU_NAME}"
  if [[ "${GPU_NAME}" != *"H20"* ]]; then
    echo "[error] detected GPU is not H20."
    echo "Set REQUIRE_H20=0 to bypass this guard if needed."
    exit 1
  fi
fi

if [[ ! -f "huggingface_token.txt" ]]; then
  echo "[error] missing huggingface_token.txt in repo root."
  echo "Please put your Hugging Face token in ${REPO_ROOT}/huggingface_token.txt"
  exit 1
fi

echo "[step 1/5] prepare datasets"
python script/reproduce/download_data.py --repo-root "${REPO_ROOT}"

echo "[step 2/5] booster alignment training"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path PKU-Alignment/BeaverTails_safe \
  --bf16 True \
  --output_dir "${ALIGN_CKPT}" \
  --num_train_epochs "${ALIGN_EPOCHS}" \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 0 \
  --learning_rate "${ALIGN_LR}" \
  --weight_decay 0.1 \
  --warmup_ratio 0 \
  --lr_scheduler_type "constant" \
  --logging_steps 10 \
  --tf32 True \
  --cache_dir "${CACHE_DIR}" \
  --optimizer booster \
  --sample_num "${ALIGN_SAMPLE_NUM}" \
  --bad_sample_num "${BAD_SAMPLE_NUM}" \
  --lamb "${LAMB}" \
  --alpha "${ALPHA}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --eval_steps 5000

echo "[step 3/5] harmful fine-tuning on SST2 setting"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --lora_folder "${ALIGN_CKPT}" \
  --data_path PKU-Alignment/BeaverTails_dangerous \
  --bf16 True \
  --output_dir "${FT_CKPT}" \
  --num_train_epochs "${FINETUNE_EPOCHS}" \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 0 \
  --learning_rate "${FINETUNE_LR}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "constant" \
  --logging_steps 10 \
  --tf32 True \
  --eval_steps 2000 \
  --cache_dir "${CACHE_DIR}" \
  --optimizer normal \
  --evaluation_strategy "steps" \
  --sample_num "${FINETUNE_SAMPLE_NUM}" \
  --poison_ratio "${POISON_RATIO}" \
  --label_smoothing_factor 0 \
  --benign_dataset data/sst2.json \
  --bad_sample_num "${BAD_SAMPLE_NUM}" \
  --lamb "${LAMB}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --alternating single_lora

echo "[step 4/5] poison safety evaluation"
cd poison/evaluation
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python pred.py \
  --lora_folder "../../${FT_CKPT}" \
  --model_folder "${MODEL_PATH}" \
  --output_path "../../${POISON_OUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python eval_sentiment.py \
  --input_path "../../${POISON_OUT}"

echo "[step 5/5] sst2 task evaluation"
cd ../../sst2
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python pred_eval.py \
  --lora_folder "../${FT_CKPT}" \
  --model_folder "${MODEL_PATH}" \
  --output_path "../${SST2_OUT}"

cd "${REPO_ROOT}"
echo "[done] all stages finished"
echo "[result] alignment ckpt: ${ALIGN_CKPT}"
echo "[result] finetune ckpt: ${FT_CKPT}"
echo "[result] poison eval: ${POISON_OUT}_sentiment_eval.json"
echo "[result] sst2 eval: ${SST2_OUT}"
