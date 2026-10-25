# nuScenes 上的 V-JEPA 预训练：数据与训练管线

[English](README.md)

本仓库提供在单张 12GB GPU 上、以 nuScenes 前视相机视频对 [V-JEPA](https://github.com/facebookresearch/jepa) 进行自监督预训练的数据与训练管线。

> **状态。** 管线已在 nuScenes mini 上完成端到端验证，计划中的四组对照实验尚未开展。现有结果仅说明预处理与训练环节运行正常，不对表征质量作任何结论。

| | |
|---|---|
| 训练代码 | [facebookresearch/jepa](https://github.com/facebookresearch/jepa)，提交 `51c59d5`（需单独 clone，不随本仓库发布） |
| 数据 | [nuScenes](https://www.nuscenes.org/) v1.0-mini，CAM_FRONT |
| 不包含 | nuScenes 数据及其生成的片段（nuScenes 使用条款不允许再分发）、模型权重 |

## 仓库结构

```
configs/vits16_nuscenes.yaml    训练配置（由官方 vitl16 配置修改而来）
scripts/preprocess_clips.py     CAM_FRONT 帧 → 2 秒 H.264 片段及 V-JEPA CSV 索引
scripts/train.sh                单卡启动脚本
requirements-lock.txt           所报告运行使用的依赖版本
results/overfit_loss.{csv,png}  验证运行的逐步损失
```

## 管线

1. **预处理**（`scripts/preprocess_clips.py`）。对每个场景，沿 `sample_data` 的 `prev`/`next` 链接按时间顺序收集全部 CAM_FRONT 帧（关键帧与 sweeps，12Hz）。断言检查时间戳严格递增、帧间隔中位数约为 83ms。每 24 帧（2 秒）编码为一个 H.264 片段，并按 V-JEPA 要求的空格分隔 CSV 格式建立索引。
2. **训练。** 使用 `configs/vits16_nuscenes.yaml` 运行官方 V-JEPA 训练代码（`app.main`，app `vjepa`）。

### 相对官方 `vitl16.yaml` 的配置改动

masking、EMA、学习率与 weight decay 调度均保持官方默认值，仅修改单卡、小数据集训练所需的设置：

| 设置 | 官方 vitl16 | 本配置 |
|---|---|---|
| 节点 × GPU | 16 × 8 | 1 × 1 |
| 模型 | ViT-L/16 | ViT-S/16 |
| `num_frames` / `sampling_rate` | 16 / 4 | 4 / 2（12Hz 下采样帧间隔约 170ms） |
| Batch size | 24 | 8 |
| Epochs / 每 epoch 迭代数 / warm-up | 300 / 300 / 40 | 10 / 100 / 1 |
| `num_workers` | 12 | 4 |

## nuScenes mini 验证运行

- **数据。** nuScenes mini 的 10 个场景共生成 92 个 2 秒片段。
- **训练。** 1,000 步（10 epoch × 100 次迭代），约 45 分钟，显存约 3GB。
- **结果。** 损失由 0.874 降至 0.341（25 步滑动平均），未出现发散或 NaN。`reg-loss` 列为 predictor 输出标准差的监控量，仅记录、不参与优化（`reg_coeff: 0`），由 0.85 变为 0.80，未见表征塌缩迹象。

![验证运行的训练损失](results/overfit_loss.png)

在 92 个片段上训练 1,000 步属于过拟合检验。损失下降表明数据加载、模型与优化器协同工作正常，并不表明模型学到了可迁移的表征。

## 路线图

- [x] nuScenes mini 上的预处理与训练管线
- [ ] 基于 nuScenes trainval（仅相机数据）生成片段并进行完整训练
- [ ] 计划中的四组对照实验

## 复现步骤

环境要求：Python 3.10、NVIDIA GPU、ffmpeg。

```bash
# 1. 依赖（numpy 与 opencv 须在同一条 pip 命令中安装，见「说明」）
pip install -r requirements-lock.txt

# 2. 上游 V-JEPA 代码（本仓库使用的提交）
git clone https://github.com/facebookresearch/jepa.git jepa
git -C jepa checkout 51c59d5

# 3. 从 nuscenes.org 下载 v1.0-mini，解压至 data/nuscenes/，然后生成片段
python scripts/preprocess_clips.py

# 4. 训练
bash scripts/train.sh
```

配置中的路径相对于 `jepa/` 目录，`scripts/train.sh` 会将其设为工作目录。

## 说明

所报告的运行使用 RTX 4080 Laptop GPU（12GB），Windows + WSL2（Ubuntu），ffmpeg 6.1.1。

- **PyTorch 版本。** 在该系统上，torch 2.13.0+cu130 在训练开始时报 `CUDA error: unknown error`，2.12.1 运行正常。两种情况下 `torch.cuda.is_available()` 均返回 `True`；在 GPU 上执行一次矩阵乘法是更可靠的检验方式。
- **numpy/opencv 依赖冲突。** V-JEPA 的依赖会安装 numpy 2.x，与 nuscenes-devkit（要求 numpy < 2）不兼容；而 opencv ≥ 4.12 要求 numpy ≥ 2。在同一条 pip 命令中安装 numpy 1.26.4 与 opencv-python 4.11.0.86 可解决该冲突。
- **训练日志。** 官方训练代码每次启动都向同一 CSV 文件追加内容并重复写入表头。`results/overfit_loss.csv` 仅保留最后一次完整运行。
- **WSL。** WSL 在最后一个会话关闭后会停止虚拟机，后台进程随之终止。长时间训练期间须保持一个会话。

## 引用

BibTeX 见 [英文 README](README.md#citation)。

## 许可证

`configs/vits16_nuscenes.yaml` 由官方 facebookresearch/jepa 配置修改而来，原配置采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可。nuScenes 数据的使用受 [nuScenes 使用条款](https://www.nuscenes.org/terms-of-use) 约束。
