from ultralytics import YOLOv10
model = YOLOv10()
model.quantize( data='coco.yaml', epochs=1, batch=12, imgsz=640)