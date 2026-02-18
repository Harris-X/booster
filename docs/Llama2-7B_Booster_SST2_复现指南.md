# Llama2-7B Booster（SST2）复现指南（中文）

本文档给出从环境准备、数据下载、Booster 对齐训练、微调到测试评估的完整步骤。

## 1. 机器与前置条件

- 系统：Linux 远程服务器
- GPU：默认要求 **H20**（脚本会自动检查，可通过环境变量关闭）
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

> 默认是 `nohup` 后台运行，即使远程终端断开任务也会继续执行。脚本启动后会打印 `pid` 和日志文件路径。

## 6. 脚本内部做了什么（完整流水线）

脚本按顺序执行以下 5 步：

1. **数据准备**
   - 下载 `data/beavertails_with_refusals_train.json`
   - 构建 `data/sst2.json`（调用 `sst2/build_dataset.py`）

2. **Booster 对齐训练（alignment）**
   - `optimizer=booster`
    - 对齐学习率默认 `1e-3`（按 T-Vaccine 共同参数覆盖）
    - LoRA 默认 `r=8, alpha=4`（按 T-Vaccine 共同参数覆盖）
   - 输出目录：
       - `ckpt/Llama-2-7b-hf_smooth_5_0.1_200_2000`

3. **有害微调（fine-tuning）**
   - 基于上一步 LoRA (`--lora_folder`)
   - `optimizer=normal`
   - 默认 poison ratio = `0.1`
    - 微调学习率默认 `1e-5`（与 T-Vaccine 一致）
   - 输出目录：
       - `ckpt/sst2/Llama-2-7b-hf_smooth_f_5_0.1_0.1_1000_200_2000`

4. **Poison 安全评测**
   - 执行：`poison/evaluation/pred.py` + `eval_sentiment.py`
   - 结果路径：
     - `data/poison/sst2/...`
     - `data/poison/sst2/..._sentiment_eval.json`

5. **SST2 任务评测**
   - 执行：`sst2/pred_eval.py`
   - 结果路径：
     - `data/sst2/...`

## 7. 如何改参数运行

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
- `POISON_RATIO`（默认：`0.1`）
- `ALIGN_EPOCHS`（默认：`20`）
- `FINETUNE_EPOCHS`（默认：`20`）
- `ALIGN_SAMPLE_NUM`（默认：`2000`，与 T-Vaccine 安全数据量 $D_a=2000$ 对齐）
- `FINETUNE_SAMPLE_NUM`（默认：`1000`）
- `BAD_SAMPLE_NUM`（默认：`200`，按 T-Vaccine 的有害样本规模映射）
- `ALIGN_LR`（默认：`1e-3`）
- `FINETUNE_LR`（默认：`1e-5`）
- `LORA_R`（默认：`8`）
- `LORA_ALPHA`（默认：`4`）
- `LAMB`（默认：`5`）
- `ALPHA`（默认：`0.1`）
- `REQUIRE_H20`（默认：`1`，必须检测到 H20）
- `ALLOW_NON_LLAMA2`（默认：`0`，仅允许 Llama2-7B）
- `ENABLE_NOHUP`（默认：`1`，自动后台运行）

## 8. 参数映射规则（你这次要求）

本脚本按“**若 Booster 与 T-Vaccine 都有该参数，则使用 T-Vaccine 值；否则保持 Booster 默认**”执行，当前映射如下：

- 对齐阶段 `learning_rate`：`1e-3`（覆盖原 `5e-4`）
- 微调阶段 `learning_rate`：`1e-5`（本来一致）
- `batch_size`：`10`（本来一致）
- `num_train_epochs`：`20`（本来一致）
- `poison_ratio`：`0.1`（本来一致）
- LoRA：`r=8, alpha=4`（通过 `train.py` 新增参数支持）
- 安全对齐样本：`ALIGN_SAMPLE_NUM=2000`（映射 T-Vaccine 的 $D_a=2000$，原 Booster 默认 5000）
- 有害样本规模：`BAD_SAMPLE_NUM=200`（映射 T-Vaccine 的 $N_h=200$）
- 对齐总步数：2000/10 × 20 = **4000 步**（T-Vaccine 一致）
- 微调总步数：1000/10 × 20 = **2000 步**（T-Vaccine 一致）

T-Vaccine 特有但 Booster 当前代码无对应实现的参数（例如 `K=200, γ=8, ρ=3` 的层采样/扰动机制），不会强行注入，以免破坏 Booster 训练逻辑。

## 9. 从哪个路径运行？

建议始终在**仓库根目录**运行：

```bash
cd /path/to/Booster
bash script/reproduce/run_llama2_booster_sst2.sh
```

脚本内部会自动切换到正确子目录执行训练和评测。

## 10. 常见问题

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
