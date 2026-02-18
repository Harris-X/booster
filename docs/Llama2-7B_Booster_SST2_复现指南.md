# Llama2-7B Booster（SST2）复现指南（中文）

本文档给出从环境准备、数据下载、Booster 对齐训练、微调到测试评估的完整步骤。

## 1. 机器与前置条件

- 系统：Linux 远程服务器
- GPU：建议单卡 80G（例如 H100/A100），或自行调小 batch
- Python：3.9（与仓库环境一致）
- 代码目录：以下示例假设仓库根目录为 `/path/to/Booster`

## 2. 获取代码并进入目录

```bash
git clone https://github.com/git-disl/Booster.git
cd Booster
```

## 3. 配置 Hugging Face Token（必须）

Llama2 是 gated repo，必须先获得 `meta-llama/Llama-2-7b-hf` 访问权限。

将 token 写入仓库根目录文件：

```bash
echo "<YOUR_HF_TOKEN>" > huggingface_token.txt
```

## 4. 创建环境并安装依赖

仓库里给了两套依赖定义，建议优先使用 conda：

```bash
conda env create -f booster.yml
conda activate vaccine
pip install -r booster_pip.txt
```

> 说明：原始脚本里出现过 `hts` 环境名，这是作者集群的本地命名；本指南统一使用 `booster.yml` 中的环境名 `vaccine`。

## 5. 一键复现脚本

已新增脚本：

- `script/reproduce/run_llama2_booster_sst2.sh`
- `script/reproduce/download_data.py`

先赋予执行权限：

```bash
chmod +x script/reproduce/run_llama2_booster_sst2.sh
```

直接运行（默认就是 Llama2-7B、Booster 对齐 + SST2 场景）：

```bash
bash script/reproduce/run_llama2_booster_sst2.sh
```

> 该脚本默认 `RUN_IN_BACKGROUND=1`，会自动用 `nohup` 后台启动；即使 SSH 断开，任务也会继续运行。
>
> 脚本会优先使用本地模型目录：`/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf`（可用 `MODEL_PATH` 或 `LOCAL_MODEL_ROOT` 覆盖）。
>
> 脚本默认设置 HF 镜像：`HF_ENDPOINT=https://hf-mirror.com`（可自行覆盖）。
>
> 脚本默认关闭 `hf_transfer`：`HF_HUB_ENABLE_HF_TRANSFER=0`，避免环境未安装 `hf_transfer` 时出现告警。

## 6. 与 T-Vaccine 的参数对齐规则

本仓库已按你的要求应用规则：

- **若 Booster 与 T-Vaccine 都有该参数**：默认改为 T-Vaccine 值
- **若 Booster 没有该参数**：保持 Booster 默认行为

当前已对齐（重叠）参数如下：

- `ALIGN_EPOCHS=20`
- `FINETUNE_EPOCHS=20`
- `BATCH_SIZE=10`
- `ALIGN_LR=1e-3`
- `FINETUNE_LR=1e-5`
- `POISON_RATIO=0.1`
- `ALIGN_SAMPLE_NUM=2000`
- `BAD_SAMPLE_NUM=200`
- `RHO=3`
- `LORA_RANK=8`
- `LORA_ALPHA=4`

未重叠/算法专有参数（如 T-Vaccine 的层采样数量、采样频率等）保持 Booster 默认，不强行映射。

## 7. 脚本内部做了什么（完整流水线）

脚本按顺序执行以下 5 步：

1. **数据准备**
   - 下载 `data/beavertails_with_refusals_train.json`
   - 构建 `data/sst2.json`（调用 `sst2/build_dataset.py`）

2. **Booster 对齐训练（alignment）**
   - `optimizer=booster`
   - 输出目录：
     - `ckpt/Llama-2-7b-hf_smooth_5_0.1_5000_5000`

3. **有害微调（fine-tuning）**
   - 基于上一步 LoRA (`--lora_folder`)
   - `optimizer=normal`
   - 默认 poison ratio = `0.1`
   - 输出目录：
     - `ckpt/sst2/Llama-2-7b-hf_smooth_f_5_0.1_0.1_1000_5000_5000`

4. **Poison 安全评测**
   - 执行：`poison/evaluation/pred.py` + `eval_sentiment.py`
   - 结果路径：
     - `data/poison/sst2/...`
     - `data/poison/sst2/..._sentiment_eval.json`

5. **SST2 任务评测**
   - 执行：`sst2/pred_eval.py`
   - 结果路径：
     - `data/sst2/...`

## 8. 如何改参数运行

脚本支持通过环境变量覆写参数，例如：

```bash
MODEL_PATH="meta-llama/Llama-2-7b-hf" \
POISON_RATIO=0.2 \
CUDA_VISIBLE_DEVICES=0 \
ALIGN_EPOCHS=20 \
FINETUNE_EPOCHS=20 \
bash script/reproduce/run_llama2_booster_sst2.sh
```

常用可配项：

- `MODEL_PATH`（默认：`meta-llama/Llama-2-7b-hf`）
- `LOCAL_MODEL_ROOT`（默认：`/data_nvme1n1/xieqiuhao/tjy/downloaded_models`）
- `MODEL_PATH`（默认优先本地 `LOCAL_MODEL_ROOT/Llama-2-7b-hf`，不存在时回退到 `meta-llama/Llama-2-7b-hf`）
- `POISON_RATIO`（默认：`0.1`）
- `ALIGN_EPOCHS`（默认：`20`）
- `FINETUNE_EPOCHS`（默认：`20`）
- `ALIGN_SAMPLE_NUM`（默认：`2000`）
- `FINETUNE_SAMPLE_NUM`（默认：`1000`）
- `BAD_SAMPLE_NUM`（默认：`200`）
- `BATCH_SIZE`（默认：`10`）
- `ALIGN_LR`（默认：`1e-3`）
- `FINETUNE_LR`（默认：`1e-5`）
- `RHO`（默认：`3`）
- `LORA_RANK`（默认：`8`）
- `LORA_ALPHA`（默认：`4`）
- `RUN_IN_BACKGROUND`（默认：`1`，自动 `nohup` 后台运行）
- `HF_ENDPOINT`（默认：`https://hf-mirror.com`）
- `HF_HUB_ENABLE_HF_TRANSFER`（默认：`0`）
- `LAMB`（默认：`5`）
- `ALPHA`（默认：`0.1`）

## 9. 查看日志与停止任务

后台启动后，日志和 pid 文件在：

- `logs/reproduce/booster_llama2_sst2_*.log`
- `logs/reproduce/booster_llama2_sst2_*.pid`

查看实时日志：

```bash
tail -f logs/reproduce/booster_llama2_sst2_*.log
```

停止任务（示例）：

```bash
kill $(cat logs/reproduce/booster_llama2_sst2_*.pid)
```

## 10. 从哪个路径运行？

建议始终在**仓库根目录**运行：

```bash
cd /path/to/Booster
bash script/reproduce/run_llama2_booster_sst2.sh
```

脚本内部会自动切换到正确子目录执行训练和评测。

## 11. 常见问题

1. **报错无法下载 Llama2 权重**
   - 检查是否通过了 `meta-llama/Llama-2-7b-hf` 访问申请
   - 检查 `huggingface_token.txt` 是否是有效 token

2. **OOM（显存不足）**
   - 降低 `per_device_train_batch_size`
   - 或降低 `model_max_length`、`sample_num`、`num_train_epochs`

3. **下载数据失败**
   - 手动下载：
     - https://huggingface.co/datasets/anonymous4486/booster_dataset/blob/main/beavertails_with_refusals_train.json
   - 放到：`data/beavertails_with_refusals_train.json`
