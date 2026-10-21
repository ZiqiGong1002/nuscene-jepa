# V-JEPA on nuScenes: data and training pipeline

[中文版](README.zh-CN.md)

A data and training pipeline for self-supervised pretraining of [V-JEPA](https://github.com/facebookresearch/jepa) on nuScenes front-camera video, on a single 12 GB GPU.

> **Status:** the pipeline runs end to end on nuScenes mini. The planned four-way comparison experiments have not started. So far this repository only shows that the preprocessing → training chain works; it makes no claim about representation quality.

## Contents

```
configs/vits16_nuscenes.yaml   training config (adapted from the official vitl16 config)
scripts/preprocess_clips.py    nuScenes CAM_FRONT frames → 2 s h264 clips + V-JEPA CSV index
scripts/train.sh               single-GPU launch script
requirements-lock.txt          package versions used for the run below
results/overfit_loss.{csv,png} per-step loss of the sanity run
```

**Not included:** nuScenes data and any clips derived from it (the nuScenes terms of use do not allow redistribution), and model weights. The V-JEPA code is not vendored; clone it separately (see below).

## Pipeline

1. **Preprocessing** (`scripts/preprocess_clips.py`). For each scene, collect all CAM_FRONT frames (keyframes and sweeps, 12 Hz) in order by following the `sample_data` `prev`/`next` chain. Assertions check that timestamps strictly increase and that the median frame gap is about 83 ms. Every 24 frames (2 s) are encoded as one h264 clip, and the clip list is written as the space-separated CSV V-JEPA expects.
2. **Training.** The official V-JEPA training code (`app.main`, app `vjepa`), run with `configs/vits16_nuscenes.yaml`.

### Config changes versus the official `vitl16.yaml`

Masking, EMA, learning-rate and weight-decay schedules keep the official defaults. Only what single-GPU training on a small dataset needs was changed:

| Setting | Official vitl16 | This config |
|---|---|---|
| nodes × GPUs | 16 × 8 | 1 × 1 |
| model | ViT-L/16 | ViT-S/16 |
| num_frames / sampling_rate | 16 / 4 | 4 / 2 (≈170 ms between sampled frames at 12 Hz) |
| batch size | 24 | 8 |
| epochs / iterations per epoch / warmup | 300 / 300 / 40 | 10 / 100 / 1 |
| num_workers | 12 | 4 |

## Getting started

Requirements: Python 3.10, an NVIDIA GPU, ffmpeg.

```bash
# 1. dependencies. numpy and opencv must be installed in the same pip command (see Notes)
pip install -r requirements-lock.txt

# 2. upstream V-JEPA code, at the commit used here
git clone https://github.com/facebookresearch/jepa.git jepa
git -C jepa checkout 51c59d5

# 3. download nuScenes v1.0-mini from nuscenes.org, extract it to data/nuscenes/, then
python scripts/preprocess_clips.py

# 4. train
bash scripts/train.sh
```

The config uses paths relative to `jepa/`, which `scripts/train.sh` makes the working directory.

## Sanity run on nuScenes mini

- **Data:** the 10 mini scenes give 92 two-second clips.
- **Training:** 1000 steps (10 epochs × 100 steps), about 45 minutes, about 3 GB of GPU memory.
- **Result:** loss fell from 0.874 to 0.341 (25-step moving average), with no divergence or NaNs. The `reg-loss` column in the CSV, a predictor-output std monitor that is logged but not trained on (`reg_coeff: 0`), went from 0.85 to 0.80, so there was no sign of collapse.

![Training loss of the sanity run](results/overfit_loss.png)

Training 1000 steps on 92 clips is deliberately an **overfitting sanity check**. The falling loss shows that data loading, model and optimizer work together. It does not show that the model learned transferable representations.

## Roadmap

- [x] Preprocessing and training pipeline on nuScenes mini
- [ ] Regenerate clips from nuScenes trainval (camera data only) and train at full scale
- [ ] Planned four-way comparison experiments

## Notes

Environment of the run: RTX 4080 Laptop GPU 12 GB, Windows + WSL2 (Ubuntu), ffmpeg 6.1.1. Problems worth knowing about:

- **torch version.** On this machine torch 2.13.0+cu130 failed with `CUDA error: unknown error` as soon as training started; 2.12.1 works. `torch.cuda.is_available()` returned True in both cases, so run a real matmul before trusting a torch install.
- **numpy / opencv conflict.** V-JEPA's dependencies pull numpy 2.x, which breaks nuscenes-devkit (needs numpy < 2), while opencv ≥ 4.12 needs numpy ≥ 2. Installing numpy 1.26.4 and opencv-python 4.11.0.86 in a single pip command resolves it.
- **Training CSV.** The official training code appends to the same CSV on every launch and repeats the header. `results/overfit_loss.csv` keeps only the last complete run.
- **WSL.** WSL shuts down its VM after the last session closes, which also kills background training. Keep a session open during long runs.

## Acknowledgements and citation

The training code is [facebookresearch/jepa](https://github.com/facebookresearch/jepa); the data is [nuScenes](https://www.nuscenes.org/). If you build on this, please cite:

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

`configs/vits16_nuscenes.yaml` is adapted from the official facebookresearch/jepa config, which is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Use of nuScenes data is governed by the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).
