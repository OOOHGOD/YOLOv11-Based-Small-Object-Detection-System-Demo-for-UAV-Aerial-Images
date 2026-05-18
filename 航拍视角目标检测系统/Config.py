#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径 (改为相对路径，适配当前项目结构)
model_path = r'models\best.pt'

names = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor'
}

CH_names = [
    '行人',
    '人群',
    '自行车',
    '小汽车',
    '面包车',
    '卡车',
    '三轮车',
    '带篷三轮车',
    '公交车',
    '摩托车'
]
