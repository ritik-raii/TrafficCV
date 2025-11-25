# Traffic Counter MVP

A modular, CPU-optimized traffic counting system using YOLOv8 and ByteTrack. Designed for offline processing of video files on low-cost hardware.

## Features

- **CPU-only Inference**: Optimized for environments without NVIDIA GPUs.
- **Object Tracking**: Uses ByteTrack (via Ultralytics) to track vehicles across frames.
- **Line Crossing Counting**: Counts vehicles crossing a user-defined line.
- **Modular Design**: Easy to swap models, adjust configuration, or extend logic.
- **Batch Processing**: Process multiple videos in sequence.
- **Exports**: Generates annotated videos, count CSVs, and track data CSVs.

## Project Structure

```
traffic-counter-mvp/
├── configs/
│   └── config.yaml       # Configuration (weights, thresholds, counting line)
├── data/
│   └── raw_videos/       # Input videos
├── exports/
│   ├── annotated_videos/ # Output videos with bounding boxes and counts
│   └── counts/           # Output CSV files
├── models/               # Place your trained .pt models here
├── src/                  # Source code
│   ├── infer.py          # Single video inference script
│   ├── batch_infer.py    # Batch processing script
│   ├── counter.py        # Counting logic
│   └── ...
├── requirements.txt      # Python dependencies
└── README.md
```

## Quickstart

### 1. Environment Setup

It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note on PyTorch:** The `requirements.txt` relies on `ultralytics` to install a compatible PyTorch version. If you are on a CPU-only machine and want to ensure the lightest installation, you can install the CPU version of PyTorch manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Configuration

Edit `configs/config.yaml` to set your preferences:
- `weights`: Path to your YOLOv8 model (e.g., `models/yolov8n.pt`).
- `count_line`: Coordinates `[[x1, y1], [x2, y2]]` for the counting line.
- `class_names`: List of class names corresponding to your model's classes.

### 3. Running Inference

**Single Video:**

```bash
python src/infer.py --input data/raw_videos/sample.mp4 --out exports/counts/ --config configs/config.yaml
```

**Batch Processing:**

```bash
python src/batch_infer.py --input_dir data/raw_videos/ --output_dir exports/counts/ --config configs/config.yaml
```

**Custom Model:**

To use a custom trained model, place the `.pt` file in `models/` and either update `config.yaml` or pass the `--weights` flag:

```bash
python src/infer.py --input data/raw_videos/sample.mp4 --weights models/custom_best.pt
```

## Counting Logic

The system uses a "line crossing" algorithm.
1.  **Detection & Tracking**: YOLOv8 detects vehicles, and ByteTrack assigns a unique ID to each vehicle across frames.
2.  **Centroid Calculation**: The center point of each vehicle's bounding box is calculated.
3.  **Crossing Check**: The system monitors the position of the centroid relative to the defined `count_line`. When a vehicle's centroid crosses from one side of the line to the other, it is counted.

**Adjusting the Line:**
Open a frame of your video and determine the pixel coordinates for the start and end of your desired counting line. Update `count_line` in `configs/config.yaml`.

## Validation

To validate the accuracy:
1.  Run inference on a short video clip (1-5 minutes).
2.  Watch the `exports/annotated_videos/` output.
3.  Manually count the vehicles crossing the line.
4.  Compare with the generated CSV report in `exports/counts/`.

## CPU Optimization

The code is explicitly configured to force CPU execution (`device='cpu'` and `CUDA_VISIBLE_DEVICES=""`). This ensures stability on non-GPU machines.
