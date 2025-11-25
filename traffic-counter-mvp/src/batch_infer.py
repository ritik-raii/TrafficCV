import argparse
import os
import subprocess
import sys
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Traffic Counter Inference")
    parser.add_argument("--input_dir", type=str, default="data/raw_videos", help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="exports/counts", help="Directory to save outputs")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    # Find video files
    extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob(os.path.join(args.input_dir, ext)))
    
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} videos to process.")

    python_executable = sys.executable
    infer_script = os.path.join("src", "infer.py")

    for video_path in video_files:
        print(f"\nProcessing {video_path}...")
        
        cmd = [
            python_executable, infer_script,
            "--input", video_path,
            "--out", args.output_dir,
            "--config", args.config
        ]
        
        if args.weights:
            cmd.extend(["--weights", args.weights])
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_path}: {e}")
            continue

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()
