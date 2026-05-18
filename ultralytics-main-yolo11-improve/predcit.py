from ultralytics import YOLO

# 1. 加载模型（官方模型 or 你的正常训练模型）
model = YOLO(r"E:\2026\ultralytics_drone\ultralytics-main-yolo11-improve\runs\detect\train11s_NeckPPA\weights\best.pt")  # 官方模型
# model = YOLO("best.pt")    # 你自己正常训练的模型（未修改网络结构）

# 2. 预测一张图片
results = model(r"E:\2026\ultralytics_drone\datasets\VOC2007\JPEGImages\0000001_02999_d_0000005.jpg")  # 换成你的图片路径

# 3. 打印结果
print("检测完成！")
results[0].show()  # 显示图片