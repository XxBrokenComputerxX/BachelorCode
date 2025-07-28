import json
import os

input_path = r"path/to/instances_val2017.json" # File in the COCO Val2017 dataset http://images.cocodataset.org/zips/val2017.zip
output_path = r"./labels" 

# Load COCO annotations
with open(input_path) as f:
    data = json.load(f)

# Create ID-to-image info map
image_map = {img['id']: img for img in data['images']}

# Create category_id to class_id mapping (0-indexed)
categories = sorted(data['categories'], key=lambda x: x['id'])
category_id_to_class_id = {cat['id']: i for i, cat in enumerate(categories)}

# Create the output directory
os.makedirs(output_path, exist_ok=True)

# Iterate through annotations
for ann in data['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]

    image = image_map[image_id]
    img_w, img_h = image['width'], image['height']

    # Convert to YOLO format
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    x_center /= img_w
    y_center /= img_h
    width /= img_w
    height /= img_h

    class_id = category_id_to_class_id[category_id]

    # Generate the full path to the label file
    file_name = os.path.splitext(image['file_name'])[0] + '.txt'
    label_path = os.path.join(output_path, file_name)

    # Write the annotation to the label file (append mode)
    with open(label_path, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
