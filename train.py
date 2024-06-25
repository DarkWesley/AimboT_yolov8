from ultralytics import YOLO

# 加载模型
model = YOLO("D:/Aimbot_for_AI/yolov8/ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml")  # 从头开始构建新模型  #训练模型（.pt权重文件）

# Use the model
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    results = model.train(data="D:/Aimbot_for_AI/yolov8/ultralytics-main/datasets/cs;go/data.yaml",
                          epochs=100,
                          batch=16,
                          imgsz=640,
                          device=0)  # 训练模型

# results = model.train(data="D:/Aimbot_for_AI/yolov8/ultralytics-main/datasets/cs/custom_data.yaml", epochs=100, batch=4, imgsz=640, device=0)  # 训练模型