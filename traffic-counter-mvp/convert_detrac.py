import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# This script assumes it is running inside 'traffic-counter-mvp'
# and the DETRAC folders are in the parent directory 'TrafficCV'

# Get the parent directory (C:\Users\HP\Desktop\PROJECTS\TrafficCV)
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(current_dir) 

# Define the expected paths for your folders
images_source_dir = os.path.join(base_path, "DETRAC-IMAGES")
xml_source_dir = os.path.join(base_path, "DETRAC-Train-Annotations-XML")

# Output directory for the ready-to-use YOLO dataset
output_dir = os.path.join(current_dir, "processed_dataset")

# UA-DETRAC Image Size is fixed
IMG_WIDTH = 960
IMG_HEIGHT = 540

# Class mapping (UA-DETRAC classes -> YOLO IDs)
class_map = {
    'car': 0,
    'bus': 1,
    'van': 2,
    'others': 3
}

def convert_box(box):
    # Convert from (left, top, width, height) to (center_x, center_y, w, h) normalized
    x_center = (box[0] + box[2] / 2) / IMG_WIDTH
    y_center = (box[1] + box[3] / 2) / IMG_HEIGHT
    w = box[2] / IMG_WIDTH
    h = box[3] / IMG_HEIGHT
    return (x_center, y_center, w, h)

def process_dataset():
    # 1. Setup Directories
    images_train_dir = os.path.join(output_dir, "images", "train")
    labels_train_dir = os.path.join(output_dir, "labels", "train")
    
    # Clean up old run if exists
    if os.path.exists(output_dir):
        print(f"Cleaning up old data at {output_dir}...")
        shutil.rmtree(output_dir)
        
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)

    print(f"\n--- CHECKING PATHS ---")
    print(f"1. Looking for Images at: {images_source_dir}")
    print(f"2. Looking for XMLs at:   {xml_source_dir}")
    
    if not os.path.exists(images_source_dir):
        print(f"\n[ERROR] Image folder not found!")
        print(f"Please move 'DETRAC-IMAGES' to: {base_path}")
        return
    if not os.path.exists(xml_source_dir):
        print(f"\n[ERROR] XML folder not found!")
        print(f"Please move 'DETRAC-Train-Annotations-XML' to: {base_path}")
        return

    print("\n--- STARTING CONVERSION ---")
    
    # Get all XML files
    xml_files = [f for f in os.listdir(xml_source_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} sequences to process.")
    
    # Process each XML sequence
    processed_count = 0
    
    for xml_file in tqdm(xml_files, desc="Processing Sequences"):
        tree = ET.parse(os.path.join(xml_source_dir, xml_file))
        root = tree.getroot()
        seq_name = root.attrib['name'] # e.g., MVI_20011
        
        # Check if the image folder for this sequence exists
        seq_img_dir = os.path.join(images_source_dir, seq_name)
        if not os.path.exists(seq_img_dir):
            # Try checking if images are directly in the folder (some versions differ)
            continue

        for frame in root.findall('frame'):
            frame_num = int(frame.attrib['num'])
            img_name = f"img{frame_num:05d}.jpg"
            src_img_path = os.path.join(seq_img_dir, img_name)
            
            if not os.path.exists(src_img_path):
                continue

            label_data = []
            target_list = frame.find('target_list')
            if target_list is not None:
                for target in target_list.findall('target'):
                    box = target.find('box')
                    attr = target.find('attribute')
                    
                    if box is not None and attr is not None:
                        v_type = attr.attrib.get('vehicle_type', 'others')
                        if v_type not in class_map: v_type = 'others'
                        class_id = class_map[v_type]

                        left = float(box.attrib['left'])
                        top = float(box.attrib['top'])
                        width = float(box.attrib['width'])
                        height = float(box.attrib['height'])

                        bbox = convert_box((left, top, width, height))
                        label_data.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

            # Only save if we have labels (skip empty frames to save space/time)
            if label_data:
                # Copy Image
                new_filename = f"{seq_name}_{img_name}"
                dst_img_path = os.path.join(images_train_dir, new_filename)
                shutil.copy(src_img_path, dst_img_path)

                # Save Label
                txt_filename = new_filename.replace('.jpg', '.txt')
                with open(os.path.join(labels_train_dir, txt_filename), 'w') as f:
                    f.write('\n'.join(label_data))
                
                processed_count += 1
                
                # LIMIT FOR TESTING: Stop after 500 images so the user can test quickly
                # Remove these lines if you want to convert the WHOLE dataset (takes long)
                if processed_count >= 500:
                    print(f"\n[INFO] Stopped after 500 images for quick testing.")
                    print("To convert everything, remove the limit in the script.")
                    return

    print(f"\nConversion Complete! {processed_count} images prepared in 'processed_dataset'.")

if __name__ == "__main__":
    process_dataset()
