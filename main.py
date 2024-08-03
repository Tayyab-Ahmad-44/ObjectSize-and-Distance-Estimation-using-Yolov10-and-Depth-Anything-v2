import cv2 as cv
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLOv10
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

yolo = YOLOv10(r"models\yolov10n_finetuned.pt")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl'  # or 'vits', 'vitb'
dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20  # 20 for indoor model, 80 for outdoor model

depthModel = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
depthModel.load_state_dict(torch.load(r'metric_depth\checkpoints\depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
depthModel = depthModel.to(DEVICE).eval()


cap = cv.VideoCapture(0)

fx = 625.19406944
fy = 625.7097461
cx = 320.84769311
cy = 242.6173792

dist = np.array([[-0.00698928, -0.02949707, 0.00183077, 0.0009398, 0.16289922]])
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def calculate_size(depth, x1, y1, x2, y2, fx, fy):
    # Calculate the real-world width and height
    width = depth * (x2 - x1) / fx
    height = depth * (y2 - y1) / fy
    return width, height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    
    # Undistort
    undistorted_frame = cv.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
    
    # Crop the image
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    results = yolo(undistorted_frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=undistorted_frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    depth = depthModel.infer_image(undistorted_frame)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Retrieve the depth value at the center of the bounding box
        distance_to_object = depth[center_y, center_x]
        
        # Calculate object size
        width, height = calculate_size(distance_to_object, x1, y1, x2, y2, fx, fy)
        print(f"Width = {width}")
        print(f"Height = {height}")
        print(f"Distance = {distance_to_object}")
        
        
        label = f"Dist: {distance_to_object:.2f}m, W: {width:.2f}m, H: {height:.2f}m"

        # Annotate the image
        cv.putText(annotated_image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('Video', annotated_image)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()