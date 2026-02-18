#!/usr/bin/env bash
set -euo pipefail

# One-click reproduction for Booster on Llama2-7B (SST2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -------------------------
# Auto-detach mode (survive SSH disconnect)
# -------------------------
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
if [[ "${RUN_IN_BACKGROUND}" == "1" && "${BOOSTER_BG_CHILD:-0}" != "1" ]]; then
  LOG_DIR="${REPO_ROOT}/logs/reproduce"
  mkdir -p "${LOG_DIR}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/booster_llama2_sst2_${TS}.log"
  PID_FILE="${LOG_DIR}/booster_llama2_sst2_${TS}.pid"

  echo "[info] launching detached job via nohup (SSH-safe)"
  BOOSTER_BG_CHILD=1 nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  CHILD_PID=$!
  echo "${CHILD_PID}" > "${PID_FILE}"
  echo "[info] started. pid=${CHILD_PID}"
  echo "[info] log=${LOG_FILE}"
  echo "[info] pid=${PID_FILE}"
  echo "[hint] tail -f ${LOG_FILE}"
  exit 0
fi

LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/data_nvme1n1/xieqiuhao/tjy/downloaded_models}"
LOCAL_LLAMA2_MODEL="${LOCAL_LLAMA2_MODEL:-${LOCAL_MODEL_ROOT}/Llama-2-7b-hf}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${LOCAL_LLAMA2_MODEL}" ]]; then
    MODEL_PATH="${LOCAL_LLAMA2_MODEL}"
  else
    MODEL_PATH="meta-llama/Llama-2-7b-hf"
  fi
else
  MODEL_PATH="${MODEL_PATH}"
fi
POISON_RATIO="${POISON_RATIO:-0.1}"
ALIGN_EPOCHS="${ALIGN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
ALIGN_SAMPLE_NUM="${ALIGN_SAMPLE_NUM:-2000}"
FINETUNE_SAMPLE_NUM="${FINETUNE_SAMPLE_NUM:-1000}"
BAD_SAMPLE_NUM="${BAD_SAMPLE_NUM:-200}"
LAMB="${LAMB:-5}"
ALPHA="${ALPHA:-0.1}"
RHO="${RHO:-3}"
CACHE_DIR="${CACHE_DIR:-cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
ALIGN_LR="${ALIGN_LR:-1e-3}"
FINETUNE_LR="${FINETUNE_LR:-1e-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-4}"

# Hugging Face mirror settings
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-${HF_ENDPOINT}}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_ENDPOINT
export HF_HUB_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER

MODEL_SHORT="$(basename "${MODEL_PATH}")"
ALIGN_CKPT="ckpt/${MODEL_SHORT}_smooth_${LAMB}_${ALPHA}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"
FT_CKPT="ckpt/sst2/${MODEL_SHORT}_smooth_f_${LAMB}_${ALPHA}_${POISON_RATIO}_${FINETUNE_SAMPLE_NUM}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"

POISON_OUT="data/poison/sst2/${MODEL_SHORT}_smooth_f_${LAMB}_${ALPHA}_${POISON_RATIO}_${FINETUNE_SAMPLE_NUM}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"
SST2_OUT="data/sst2/${MODEL_SHORT}_smooth_f_${LAMB}_${ALPHA}_${POISON_RATIO}_${FINETUNE_SAMPLE_NUM}_${BAD_SAMPLE_NUM}_${ALIGN_SAMPLE_NUM}"

echo "[info] repo root: ${REPO_ROOT}"
echo "[info] model: ${MODEL_PATH}"
echo "[info] poison ratio: ${POISON_RATIO}"
echo "[info] hf endpoint: ${HF_ENDPOINT}"
echo "[info] hf_transfer enabled: ${HF_HUB_ENABLE_HF_TRANSFER}"
echo "[info] overlap params from T-Vaccine: batch=${BATCH_SIZE}, align_lr=${ALIGN_LR}, finetune_lr=${FINETUNE_LR}, align_samples=${ALIGN_SAMPLE_NUM}, harmful_samples=${BAD_SAMPLE_NUM}, rho=${RHO}, lora_rank=${LORA_RANK}, lora_alpha=${LORA_ALPHA}"

cd "${REPO_ROOT}"

mkdir -p data ckpt ckpt/sst2 "${CACHE_DIR}" data/poison data/poison/sst2 data/sst2

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
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
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
  --rho "${RHO}" \
  --lora_r "${LORA_RANK}" \
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
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
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
  --rho "${RHO}" \
  --lora_r "${LORA_RANK}" \
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
