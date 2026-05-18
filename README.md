# 基于YOLOv11的无人机航拍图像小目标检测系统Demo
本次本科毕业设计采用PyQt5完成图形界面开发，以YOLOv11算法为基础，融合PPA注意力模块与SAHI切片推理策略，实现对VisDrone2019无人机数据集中小目标的检测。

**补充说明：** 实验数据并非完全真实可靠，实际检测性能未能达到**mAP@0.5：0.462**，该程序仅用于毕业设计考核使用。

**完整项目链接（包含数据集）：**
https://pan.baidu.com/s/18q3HCUhYEjuva-T54DFuKg?pwd=2026 提取码: 2026 
--来自百度网盘超级会员v5的分享
# YOLOv11-Based-Small-Object-Detection-System-Demo-for-UAV-Aerial-Images
This undergraduate graduation design adopts PyQt5 for GUI development. Based on YOLOv11, PPA attention module and SAHI strategy are integrated to detect small targets in VisDrone2019 UAV datasets.

**Supplement note:** The experimental data is not fully authentic, and actual performance fails to reach the displayed level. This program is mainly for graduation assessment.
---

## 项目概述

本项目是一个基于深度学习的无人机航拍图像小目标检测系统，采用 YOLOv11 作为基础检测模型，结合 PPA（PPAttention）注意力模块和 SAHI 切片推理策略，针对 VisDrone2019 无人机数据集进行优化。

### 核心特性

- **YOLOv11 基础模型**: 采用 Ultralytics YOLOv11 作为检测骨干
- **PPA 注意力模块**: 在 Backbone 和 Neck 部分集成 PPA 注意力机制，增强特征提取能力
- **SAHI 切片推理**: 使用 SAHI (Slicing Aided Hyper Inference) 策略提升小目标检测性能
- **PyQt5 图形界面**: 提供友好的可视化操作界面，支持图片、视频、摄像头检测
- **GPU/CPU 切换**: 支持显卡加速和 CPU 推理模式

---

## 项目结构

```
YOLOv11-Based-Small-Object-Detection-System-Demo-for-UAV-Aerial-Images/
├── 航拍视角目标检测系统/          # 主程序 GUI 应用
│   ├── MainProgram.py            # 主窗口程序入口
│   ├── Config.py                 # 配置文件
│   ├── detect_tools.py           # 检测工具函数
│   ├── imgTest.py                # 图片检测模块
│   ├── VideoTest.py              # 视频检测模块
│   ├── CameraTest.py             # 摄像头检测模块
│   ├── UIProgram/                # UI 相关模块
│   ├── models/                   # 模型文件目录
│   └── requirements.txt          # 依赖清单
├── ultralytics-main-yolo11-improve/  # YOLO 训练与实验代码
│   ├── train_v11.py              # 训练脚本
│   ├── val_sahi.py               # SAHI 验证脚本
│   ├── contrast_yolov11_ppa_sahi.py  # 对比实验脚本
│   └── ultralytics/              # Ultralytics 源码
├── VisDrone2019-DET-train/       # 训练集
├── VisDrone2019-DET-val/         # 验证集
├── VisDrone2019-DET-test-dev/    # 测试集
├── SSD-master/                   # SSD 对比实验代码
└── visdrone.yaml                # 数据集配置文件
```

---

## 支持的检测类别

| ID | 英文名称 | 中文名称 |
|----|----------|----------|
| 0 | pedestrian | 行人 |
| 1 | people | 人 |
| 2 | bicycle | 自行车 |
| 3 | car | 汽车 |
| 4 | van | 面包车 |
| 5 | truck | 卡车 |
| 6 | tricycle | 三轮车 |
| 7 | awning_tricycle | 带篷三轮车 |
| 8 | bus | 公交车 |
| 9 | motor | 摩托车 |

---

## 功能特点

### 1. 多模式检测
- **单图检测**: 支持 JPG、PNG、BMP 等格式图片
- **批量图片检测**: 支持文件夹批量处理，自动遍历所有图片
- **视频检测**: 支持 AVI、MP4、WMV、MKV 等格式视频
- **摄像头检测**: 支持实时摄像头画面检测

### 2. 参数可调
- **置信度阈值**: 0.0 ~ 1.0 可调
- **IOU 阈值**: 0.0 ~ 1.0 可调
- **显示标签**: 可开关检测框标签显示
- **显示原图**: 可切换原图/检测结果视图

### 3. 导航功能
- 批量图片/视频帧切换浏览
- 上一张/下一张便捷按钮

### 4. 模型管理
- 支持加载自定义 YOLO 模型文件 (.pt)
- 设备切换 (GPU/CPU)

---

## 安装依赖

```bash
cd 航拍视角目标检测系统
pip install -r requirements.txt
```

### 主要依赖

- Python >= 3.8
- PyQt5 >= 5.15.2
- ultralytics >= 8.0.0
- torch >= 2.0.0
- opencv-python >= 4.5.0
- sahi >= 0.11.0

---

## 使用方法

### 启动 GUI 程序

```bash
cd 航拍视角目标检测系统
python MainProgram.py
```

### 训练自己的模型

```bash
cd ultralytics-main-yolo11-improve
python train_v11.py
```

### 对比实验

```bash
cd ultralytics-main-yolo11-improve
python contrast_yolov11_ppa_sahi.py
```

---

## 模型配置说明

### PPA 模块消融实验配置

在 `train_v11.py` 中可选择不同的模型配置：

```python
# 基础 YOLOv11
model_yaml_path = r"ultralytics\cfg\models\v8\yoloe-v8.yaml"

# Backbone PPA
model_yaml_path = r"ultralytics\cfg\models\11\yolo11-backbone-PPA.yaml"

# Neck PPA (推荐用于小目标检测)
model_yaml_path = r"ultralytics\cfg\models\11\yolo11-neck-PPA.yaml"
```

### 模型规模对比

| 模型 | 参数数量 | FLOPs |
|------|----------|-------|
| YOLOv11n | 2.6M | 6.4G |
| YOLOv11s | 11.6M | - |
| YOLOv11-backbone-PPA | 4.8M | 8.0G |
| YOLOv11-neck-PPA | 5.4M | 11.2G |

---

## 数据集说明

本项目使用 [VisDrone2019](http://aiskyeye.com/) 无人机航拍数据集。

### 数据集划分

- **训练集**: VisDrone2019-DET-train (约 6k 张图片)
- **验证集**: VisDrone2019-DET-val (约 1k 张图片)
- **测试集**: VisDrone2019-DET-test-dev

### 数据格式

支持 VOC 格式和 YOLO 格式转换：

```bash
# VOC 转 YOLO
python datatranslatevoc.py

# 检查数据
python checkdatavoc.py
```

---

## 项目截图&演示文稿

系统提供直观的图形界面，包括：
- 主显示区域：实时显示检测结果
- 参数设置面板：置信度、IOU、设备选择
- 结果信息面板：检测数量、类别、置信度、坐标
- 操作按钮：图片、视频、摄像头、批量处理

<img width="49%" src="https://github.com/user-attachments/assets/db7fd1ad-9bb6-4e54-a5e0-ca0ba40055cd" />
<img width="49%" src="https://github.com/user-attachments/assets/df52701c-69f0-4664-8dbe-2ae4e9f89a33" />

<img width="49%" src="https://github.com/user-attachments/assets/0e8d0df2-4ab9-49f7-be35-2470e459efcd" />
<img width="49%" src="https://github.com/user-attachments/assets/bf3e7367-5b84-4e61-8e0e-cde0b48789c1" />

<img width="49%" src="https://github.com/user-attachments/assets/dc5e1614-773f-4845-8928-3316472f1132" />
<img width="49%" src="https://github.com/user-attachments/assets/8db8531b-4ae7-4054-a65c-ba4354b7ce89" />

<img width="49%" src="https://github.com/user-attachments/assets/bebfd320-4724-4e2b-abd5-87c1ca19286b" />
<img width="49%" src="https://github.com/user-attachments/assets/251b55fa-c897-4c80-9a78-7583a2a18d27" />

<img width="49%" src="https://github.com/user-attachments/assets/84ddda19-a935-496a-b13d-9fd00d7df1af" />
<img width="49%" src="https://github.com/user-attachments/assets/e9dc9975-03f5-4a2e-88bb-ecea556974ed" />

<img width="49%" src="https://github.com/user-attachments/assets/62f916b2-9801-4467-b92c-20b20bb3eebc" />
<img width="49%" src="https://github.com/user-attachments/assets/0942d245-665e-43cf-a55c-bf1215746b3a" />

<img width="49%" src="https://github.com/user-attachments/assets/cee6e7b5-5459-45e7-aa51-c571d89d4972" />
<img width="49%" src="https://github.com/user-attachments/assets/36fdf325-6558-4630-9c0c-0be3a368c7ff" />

<img width="49%" src="https://github.com/user-attachments/assets/12054138-ba9e-41a9-aa3d-d1aa3603e31a" />
---

## 注意事项

1. **数据集下载**: 完整项目（含数据集）可通过百度网盘获取，链接见原 README
2. **实验数据**: 页面显示的 mAP@0.5: 0.462 为标称性能，实际结果可能有所差异
3. **GPU 支持**: 推荐使用 NVIDIA 显卡以获得最佳性能，CUDA 11.x+ 推荐
4. **路径配置**: 首次运行可能需要根据实际环境调整模型路径

---

## 参考资源

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [SAHI - Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [VisDrone Dataset](http://aiskyeye.com/)

---

## 许可证

MIT License