# V-JEPA Pretraining on nuScenes: Data and Training Pipeline

[中文版](README.zh-CN.md)

This repository provides a data and training pipeline for self-supervised pretraining of [V-JEPA](https://github.com/facebookresearch/jepa) on nuScenes front-camera video using a single 12 GB GPU.

> **Status.** The pipeline has been verified end to end on nuScenes mini. The planned four-way comparison experiments have not yet been conducted. The current results establish only that the preprocessing and training stages function correctly; no claim is made about representation quality.

| | |
|---|---|
| Training code | [facebookresearch/jepa](https://github.com/facebookresearch/jepa), commit `51c59d5` (cloned separately, not included) |
| Data | [nuScenes](https://www.nuscenes.org/) v1.0-mini, CAM_FRONT |
| Not included | nuScenes data and derived clips (redistribution is not permitted by the nuScenes terms of use), model weights |

## Repository structure

```
configs/vits16_nuscenes.yaml    training configuration (adapted from the official vitl16 configuration)
scripts/preprocess_clips.py     CAM_FRONT frames → 2 s H.264 clips and V-JEPA CSV index
scripts/train.sh                single-GPU launch script
requirements-lock.txt           package versions used for the reported run
results/overfit_loss.{csv,png}  per-step loss of the verification run
```

## Pipeline

1. **Preprocessing** (`scripts/preprocess_clips.py`). For each scene, all CAM_FRONT frames (keyframes and sweeps, 12 Hz) are ordered by following the `prev`/`next` links of `sample_data`. Assertions verify that timestamps increase strictly and that the median inter-frame interval is approximately 83 ms. Each block of 24 frames (2 s) is encoded as one H.264 clip, and the clips are indexed in the space-separated CSV format expected by V-JEPA.
2. **Training.** The official V-JEPA training code (`app.main`, app `vjepa`) is run with `configs/vits16_nuscenes.yaml`.

### Configuration changes relative to the official `vitl16.yaml`

Masking, EMA, learning-rate and weight-decay schedules retain the official defaults. Only the settings required for single-GPU training on a small dataset were changed:

| Setting | Official vitl16 | This configuration |
|---|---|---|
| Nodes × GPUs | 16 × 8 | 1 × 1 |
| Model | ViT-L/16 | ViT-S/16 |
| `num_frames` / `sampling_rate` | 16 / 4 | 4 / 2 (≈170 ms between sampled frames at 12 Hz) |
| Batch size | 24 | 8 |
| Epochs / iterations per epoch / warm-up | 300 / 300 / 40 | 10 / 100 / 1 |
| `num_workers` | 12 | 4 |

## Verification run on nuScenes mini

- **Data.** The 10 scenes of nuScenes mini yield 92 two-second clips.
- **Training.** 1,000 steps (10 epochs × 100 iterations), approximately 45 minutes and 3 GB of GPU memory.
- **Result.** The loss decreased from 0.874 to 0.341 (25-step moving average) without divergence or NaN values. The `reg-loss` column, a monitor of the predictor-output standard deviation that is logged but not optimised (`reg_coeff: 0`), changed from 0.85 to 0.80, giving no indication of representation collapse.

![Training loss of the verification run](results/overfit_loss.png)

Training for 1,000 steps on 92 clips is intended as an overfitting check. The decreasing loss confirms that data loading, model and optimiser operate correctly together; it does not indicate that the model has learned transferable representations.

## Roadmap

- [x] Preprocessing and training pipeline on nuScenes mini
- [ ] Clip generation from nuScenes trainval (camera data only) and full-scale training
- [ ] Planned four-way comparison experiments

## Reproducibility

Requirements: Python 3.10, an NVIDIA GPU and ffmpeg.

```bash
# 1. Dependencies (numpy and opencv must be installed in a single pip command; see Notes)
pip install -r requirements-lock.txt

# 2. Upstream V-JEPA code at the commit used here
git clone https://github.com/facebookresearch/jepa.git jepa
git -C jepa checkout 51c59d5

# 3. Download nuScenes v1.0-mini from nuscenes.org, extract it to data/nuscenes/, then
python scripts/preprocess_clips.py

# 4. Training
bash scripts/train.sh
```

Paths in the configuration are relative to `jepa/`, which `scripts/train.sh` sets as the working directory.

## Notes

The reported run used an RTX 4080 Laptop GPU (12 GB) under Windows with WSL2 (Ubuntu) and ffmpeg 6.1.1.

- **PyTorch version.** On this system, torch 2.13.0+cu130 failed with `CUDA error: unknown error` at the start of training, whereas 2.12.1 runs correctly. `torch.cuda.is_available()` returned `True` in both cases; a matrix multiplication on the GPU is a more reliable check.
- **numpy/opencv dependency conflict.** The V-JEPA dependencies install numpy 2.x, which is incompatible with nuscenes-devkit (requires numpy < 2), while opencv ≥ 4.12 requires numpy ≥ 2. Installing numpy 1.26.4 and opencv-python 4.11.0.86 in a single pip command resolves the conflict.
- **Training log.** The official training code appends to the same CSV file on each launch and repeats the header. `results/overfit_loss.csv` contains only the final complete run.
- **WSL.** WSL stops its virtual machine after the last session closes, which also terminates background processes. A session must remain open during long training runs.

## Citation

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}

@inproceedings{caesar2020nuscenes,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={CVPR},
  year={2020}
}
```

## License

`configs/vits16_nuscenes.yaml` is adapted from the official facebookresearch/jepa configuration, which is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Use of nuScenes data is governed by the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).
