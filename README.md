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
