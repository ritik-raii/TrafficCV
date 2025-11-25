from ultralytics import YOLO
import os

def main():
    # Get absolute path to the yaml file to avoid path issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, 'detrac.yaml')
    
    print(f"Starting training with config: {yaml_path}")
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # We use absolute path for 'data' to be safe
    results = model.train(data=yaml_path, epochs=10, imgsz=640)

if __name__ == '__main__':
    main()
