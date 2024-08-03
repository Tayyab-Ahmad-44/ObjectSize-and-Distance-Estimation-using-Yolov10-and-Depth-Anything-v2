# Object Size and Distance Estimation using YOLO and DepthAnything v2

## Introduction

This project uses the YOLOv10 object detection model and the DepthAnything v2 depth estimation model to estimate the size and distance of objects in real-time from a video feed. The application captures video from a webcam, processes each frame to detect objects, and then estimates the depth, width, and height of the detected objects.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- PyTorch
- Supervision
- Ultralytics YOLOv10
- DepthAnything v2

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/object-size-distance-estimation.git
    cd object-size-distance-estimation
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the YOLOv10 and DepthAnything v2 model weights and place them in the appropriate directories:
    - YOLOv10: `models/yolov10n_finetuned.pt`
    - DepthAnything v2: `metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth`

## Usage

Run the main script to start the video capture and object detection:
```bash
python main.py
