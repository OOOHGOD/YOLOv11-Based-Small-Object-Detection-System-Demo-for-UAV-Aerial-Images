<div align="center">

# UAV Small-Object Detection

YOLO11, PPA, and SAHI experiments for aerial imagery.

**English** · [简体中文](README.zh-CN.md)

![YOLO11](https://img.shields.io/badge/YOLO11-111827?style=flat-square) ![SAHI](https://img.shields.io/badge/SAHI-334155?style=flat-square) ![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=flat-square) [![MIT License](https://img.shields.io/badge/License-MIT-2563EB?style=flat-square)](LICENSE)

</div>

<p align="center">
<a href="#overview">Overview</a> · <a href="#setup-and-gui">Setup and GUI</a> · <a href="#features-and-structure">Features and structure</a> · <a href="#author--license">Author &amp; License</a>
</p>

---

> **Project focus** · Educational Demo


## Overview

An undergraduate project exploring UAV small-object detection, with YOLO11/PPA experiments, SAHI sliced inference, and a PyQt5 desktop interface.


## Features and structure

| Path | Purpose |
| --- | --- |
| `UAV Aerial Perspective Small Target Detection System/` | GUI, image/video/camera handling, model configuration, and inference tools |
| `ultralytics-main-yolo11-improve/` | Vendored source, model configurations, PPA experiments, and SAHI evaluation |
| `SSD-master/`, `pytorch-ssd-master/` | Third-party SSD comparison implementations |
| `train11.py`, `train8.py`, `trainssd.py` | Additional experiments |
| `visdrone.yaml` and conversion scripts | Dataset configuration and annotation preparation |

The interface supports image, batch, video, and camera detection, confidence/IoU controls, custom models, and CPU/GPU selection. GUI inference and experiment scripts are separate entry points; inspect the selected model when comparing output.

## Setup and GUI

Create and activate an isolated environment, then run:

```bash
cd "UAV Aerial Perspective Small Target Detection System"
python -m pip install -r requirements.txt
python MainProgram.py
```

Check `Config.py` first: `model_path` defaults to `models\best.pt`; output goes under `save_data`. Supply a compatible checkpoint. A standard Ultralytics installation alone may not contain custom modules needed by experimental checkpoints.

## Data and experiments

The experiments use VisDrone2019 and ten labels: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor.

Prepare authorized data and update dataset YAMLs and local paths. The original [full-material sharing link](https://pan.baidu.com/s/18q3HCUhYEjuva-T54DFuKg?pwd=2026), code `2026`, is retained; availability and rights must be checked separately.

From the repository root:

```bash
cd ultralytics-main-yolo11-improve
python train_v11.py
```

Inspect `model_yaml_path` and `data_yaml_path`. The script contains baseline and PPA alternatives; select the active YAML deliberately rather than inferring it from the output name. Review `val_sahi.py` and `contrast_yolov11_ppa_sahi.py` for sliced inference and comparison experiments, and configure their checkpoint/data paths.

## Evidence and limitations

The [Chinese README](README.zh-CN.md) preserves original screenshots and experiment context. The original English note states that displayed experiment values were not fully authentic and actual performance did not reach the displayed level. Accordingly, those values are not presented as validated benchmark results. Independent evaluation on a documented split is needed.

## Third-party code

The pre-existing root MIT notice does not replace the vendored Ultralytics AGPL notices or bundled SSD licenses. The repository does not provide blanket MIT permission for upstream source, weights, or VisDrone data. See `LICENSE_SCOPE.md` before redistribution or combined use.

## Author & License

Author / maintainer: [OOOHGOD](https://github.com/OOOHGOD) · Original contributions: [MIT License](LICENSE)

Commercial use and modification are permitted with preservation of copyright and license notices in copies or substantial portions. Third-party code, datasets, models, and media retain their own terms; see [license scope](LICENSE_SCOPE.md).
