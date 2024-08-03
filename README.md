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

4. Download the DepthAnything v2 model from [this link](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) and place it in the `metric_depth/checkpoints/` folder.

5. Obtain the YOLOv10 model and fine-tune it on the objects that you want to find the size and distance of. Place the fine-tuned model in the `models/` folder.

## Camera Calibration

First, you need to calibrate your camera:

1. Go to the `CameraCalibration` folder and run the `get_images.py` script to capture images of a chessboard pattern.
2. After capturing the images, run the `calibration.py` script in the same folder to obtain the camera calibration matrix and other relevant parameters.
3. Update the parameters in the `main.py` script with the calibration matrix obtained from the previous step.

## Usage

Run the main script to start the video capture and object detection:
```bash
python main.py
