import argparse
import os
import cv2
import pandas as pd
import yaml
from tqdm import tqdm
from ultralytics import YOLO
import torch

from src.utils import load_config, ensure_dir
from src.counter import Counter
from src.mapping import build_class_map

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Counter Inference (CPU-only)")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out", type=str, default="exports/counts", help="Directory to save counts CSV")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, help="Path to model weights (overrides config)")
    parser.add_argument("--show", action="store_true", help="Show video during processing")
    return parser.parse_args()

def main():
    # Force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Load config
    config = load_config(args.config)
    
    # Override weights if provided
    weights_path = args.weights if args.weights else config.get("weights", "models/yolov8n.pt")
    
    # Ensure output directories
    ensure_dir(args.out)
    annotated_dir = os.path.join("exports", "annotated_videos")
    ensure_dir(annotated_dir)

    # Initialize Model
    print(f"Loading model from {weights_path}...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Counter
    line_points = config.get("count_line", [[100, 300], [1200, 300]])
    line = (tuple(line_points[0]), tuple(line_points[1]))
    counter = Counter(
        line=line,
        min_frames=config.get("min_track_frames", 2),
        min_bbox_area=config.get("min_bbox_area", 400)
    )
    
    class_names = config.get("class_names", [])
    class_map = build_class_map(class_names)

    # Video Setup
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error opening video file {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_video_path = os.path.join(annotated_dir, f"{video_basename}_annotated.mp4")
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing {args.input}...")
    print(f"Output video: {output_video_path}")
    print(f"Device: CPU")

    track_history = []

    # Inference Loop
    # Using model.track() with stream=True for generator
    results_generator = model.track(
        source=args.input,
        conf=config.get("conf_threshold", 0.35),
        iou=config.get("iou_threshold", 0.45),
        device="cpu",
        persist=True,
        stream=True,
        verbose=False,
        tracker="bytetrack.yaml" # Default tracker
    )

    pbar = tqdm(total=total_frames, desc="Processing Frames")

    for frame_idx, result in enumerate(results_generator):
        frame = result.orig_img
        
        # Draw counting line
        cv2.line(frame, line[0], line[1], (0, 255, 0), 2)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            cls = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, class_id in zip(boxes, ids, cls):
                x1, y1, x2, y2 = box
                
                # Update counter
                counter.process_detection(track_id, class_id, (x1, y1, x2, y2), frame_idx)
                
                # Record track history for CSV
                track_history.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_map.get(class_id, str(class_id)),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                # Draw bbox and label
                label = f"#{track_id} {class_map.get(class_id, str(class_id))}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw counts on screen
        counts = counter.get_counts()
        y_offset = 30
        for cls_id, count in counts.items():
            cls_name = class_map.get(cls_id, str(cls_id))
            text = f"{cls_name}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        out_vid.write(frame)
        
        if args.show:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pbar.update(1)

    pbar.close()
    out_vid.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    # Save Counts CSV
    counts_data = []
    final_counts = counter.get_counts()
    for cls_id, count in final_counts.items():
        counts_data.append({
            "class_id": cls_id,
            "class_name": class_map.get(cls_id, str(cls_id)),
            "count": count
        })
    
    counts_df = pd.DataFrame(counts_data)
    counts_csv_path = os.path.join(args.out, f"{video_basename}_counts.csv")
    counts_df.to_csv(counts_csv_path, index=False)
    print(f"Saved counts to {counts_csv_path}")

    # Save Tracks CSV
    tracks_df = pd.DataFrame(track_history)
    tracks_csv_path = os.path.join(args.out, f"{video_basename}_tracks.csv")
    tracks_df.to_csv(tracks_csv_path, index=False)
    print(f"Saved tracks to {tracks_csv_path}")

if __name__ == "__main__":
    main()
