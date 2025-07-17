from ultralytics import YOLO

# 1. 加载你训练好的权重
#    假设你的 best.pt 存在于 runs/train/exp/weights/best.pt
model = YOLO('./ls_detect_runs/train/exp2/weights/best.pt')

# 2. 设置测试集路径
test_images = '/root/autodl-fs/landslide_split/images/test'   # 测试集图片文件夹
test_labels = '/root/autodl-fs/landslide_split/labels/test'   # 如果有标签，用于评估

# 3. 推理（Inference）并保存可视化结果
#    - source: 测试集图片目录或单张图片
#    - save: 是否保存带框的图片
#    - save_dir: 结果保存目录
#    - conf: 置信度阈值（可调）
results = model.predict(
    source=test_images,
    conf=0.25,
    save=True,
    save_dir='ls_detect_runs/predict/exp',   # 推理结果目录，会自动创建
)

# results 是一个 Results 对象列表，你可以用它做二次处理：
for r in results:
    # r.boxes.xyxy: N×4 tensor, 每行是 [x1,y1,x2,y2]
    # r.boxes.conf: N tensor, 置信度
    # r.boxes.cls:  N tensor, 类别 id
    print(r.path, '->', r.boxes.shape[0], 'boxes detected')

# 4. （可选）如果有真值标签，评估指标
#    直接调用 val()，参数和 train() 类似，但只跑一次验证
metrics = model.val(
    data='./ls_test.yaml',  # data.yaml 中已指定 test 集
    imgsz=640,
    iou=0.65,
    batch=16,
    workers=16
)
