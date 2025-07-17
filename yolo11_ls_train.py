from ultralytics import YOLO

# Load a model
model = YOLO("yolo12s.pt")
# model = YOLO("yolo11n.yaml")
# Train the model
train_results = model.train(
    data="/root/autodl-tmp/yolo11/ls_uav_dataset.yaml",  # path to CrowdHuman YAML
    epochs=200,
    batch=64,
    imgsz=640,
    optimizer='AdamW',
    amp=True,
    project='ls_detect_runs/train',
    name='yolo12n'
)

# Evaluate model performance on the validation set
#metrics = model.val()

#yolo train data=/root/ultralytics-main/ultralytics/cfg/datasets/CrowdHuman.yaml model=yolo11n.yaml epochs=100 imgsz=640 batch=64 optimizer='SGD' amp=True
# Perform object detection on an image
#results = model("path/to/image.jpg")
#results[0].show()
#ultralytics-main/WiderPerson_2
# Export the model to ONNX format
#path = model.export(format="onnx")  # return path to exported model
#D:\yolo11\ultralytics-main\WiderPerson
#root/ultralytics-main
#scp -o StrictHostKeyChecking=no -r -P 42212 D:\dataset\VisDrone root@ssh.intern-ai.org.cn:ultralytics-main