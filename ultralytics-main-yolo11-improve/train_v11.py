#coding:utf-8
from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')

# 模型配置文件 骨干和颈部PPA消融实验
model_yaml_path = r"ultralytics\cfg\models\v8\yoloe-v8.yaml"
#model_yaml_path = r"ultralytics\cfg\models\11\yolo11.yaml"
#model_yaml_path = r"ultralytics\cfg\models\11\yolo11-backbone-PPA.yaml"
#model_yaml_path = r"ultralytics\cfg\models\11\yolo11-neck-PPA.yaml"
#数据集配置文件
data_yaml_path = r"ultralytics\cfg\datasets\VisDrone.yaml"
#         YOLO11 summary: 181 layers, 2,590,035 parameters, 2,590,019 gradients, 6.4 GFLOPs
#YOL011-backbone-PPA summary: 209 layers, 4,761,275 parameters, 4,761,259 gradients, 8.0 GFL0Ps
#YOL011-neck-PPA summary: 265 layers, 5,443,657 parameters, 5,443,641 gradients, 11.2 GFLOPs

if __name__ == '__main__':
    #加载预训练模型
    model = YOLO(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=100,      # 训练轮数
                          batch=8,         # batch大小
                          name='train11s_neckPPA', # 保存结果的文件夹名称
                          optimizer='SGD')  # 优化器