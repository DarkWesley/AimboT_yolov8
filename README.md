# AimboT

## 环境

本`AimboT`文件基于YOLOv8（官网https://github.com/ultralytics/ultralytics/），环境如下：

<table>
  <thead><tr><th>Windows</th><td>10 or 11(更优)</td></thead>
  <thead><tr><th>Python:</th><td>3.11及以上</td></tr></thead>
  <thead><tr><th>CUDA:</th><td>12.4</td></tr></thead>
  <thead><tr><th>TensorRT:</th><td>10.0.1</td></tr></thead>
</table>

## 实验准备

在根目录下， 打开`cmd`，输入：

```
pip install -r requirements.txt
```

若出现找不到库的情况，则需要更新`pip`：

```
python -m pip install --upgrade pip
```

## 数据集准备

可以通过`www.roboflow.com`或`kaggle`获取数据集。

本实验自带数据集位于`./datasets`目录下。

选择数据集后，查看数据集文件夹下的`.yaml`文件。以`./datasets/cs2`为例，`./datasets/cs2/csgo-kaggle.yaml`文件夹中，记录了识别目标的`name_class`。

打开`./train.py`文件，修改其中的

```
model = YOLO("YOUR-DIRECTORY/ultralytics/cfg/models/v8/yolov8s.yaml")           # 选择YOLO预训练模型

......

results = model.train(data="YOUR-DIRECTORY/datasets/cs2/csgo-kaggle.yaml",      # 选择数据集文件训练模型
                          epochs=100,
                          batch=16,
                          imgsz=640,
                          device=0)
```

训练你的模型。这里的地址可以用相对地址，但最好使用**绝对地址**。

其中，`model.train()`的参数可以根据你的实际情况修改。

训练完成后，获得`./runs`目录下的训练文件，找到权重文件`*.pt`。

## 运行

打开`./main.py`，根据你选择的数据集中的`name_class`修改一下内容：

+ `detection_modes`中的数据：`teamCT`表示你是**反恐精英**的一员，目标的`classes`数组的值设定为`*.yaml`文件中，**恐怖分子**的`t_body`和`t_head`序号；反之，`teamT`则将目标设定为`ct_body`和`ct_head`的序号；若是`Solo`，适用于你在参加**死亡竞赛**模式的情况，所有人都是你的敌人，`t/ct_body/head`都要设定进去。
+ `resize`是你的监测器小窗口的大小，任意调整，但想获得更好的效果则将其设定为与你的游戏窗口相同的比例
+ 在`mouse_move`函数中，找到写有`if distance_list[0][0] == 1 or distance_list[0][0] == 3: `的两行，将其中的`1 or 3`改为对应的`t_head`和`ct_head`序号，这样才能锁头
  
修改完毕后即可运行

### 注意事项

1. 本自瞄系统**没有**自动移动和自动转移视角，需要敌人出现在视野里才能锁上去，因此需要手动操作`WASD`移动
2. `main.py`启动后，在游戏内按`Tab`键启动锁头，按`F5`,`F6`,`F7`分别切换模式到`teamCT`,`teamT`和`Solo`