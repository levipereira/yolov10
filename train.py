from ultralytics import YOLOv10
model = YOLOv10()
model.quantize( data='coco.yaml', epochs=1, batch=18, imgsz=640)