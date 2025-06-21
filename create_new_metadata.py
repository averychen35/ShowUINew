import json
import os
from PIL import Image
import shutil

def crop_image_around_point(image_path, point, output_size_ratio=0.5):
    """
    Crop image to 1/4 area (1/2 width × 1/2 height) centered around the given point.
    
    Args:
        image_path: Path to the original image
        point: [x, y] coordinates (0-1 normalized)
        output_size_ratio: Ratio of output dimensions to original (0.5 = 1/2 each dimension = 1/4 area)
    
    Returns:
        PIL Image object of the cropped image
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Convert normalized coordinates to pixel coordinates
    center_x = int(point[0] * width)
    center_y = int(point[1] * height)
    
    # Calculate crop dimensions (1/2 of original dimensions = 1/4 area)
    crop_width = int(width * output_size_ratio)
    crop_height = int(height * output_size_ratio)
    
    # Calculate crop boundaries
    left = max(0, center_x - crop_width // 2)
    top = max(0, center_y - crop_height // 2)
    right = min(width, center_x + crop_width // 2)
    bottom = min(height, center_y + crop_height // 2)
    
    # Adjust if crop would go out of bounds
    if right - left < crop_width:
        if left == 0:
            right = min(width, left + crop_width)
        else:
            left = max(0, right - crop_width)
    
    if bottom - top < crop_height:
        if top == 0:
            bottom = min(height, top + crop_height)
        else:
            top = max(0, bottom - crop_height)
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    
    return cropped_img, (left, top, right, bottom)

def recalculate_coordinates(original_bbox, original_point, crop_bounds, original_img_size):
    """
    Recalculate bounding box and point coordinates for the cropped image.
    
    Args:
        original_bbox: [x1, y1, x2, y2] normalized coordinates
        original_point: [x, y] normalized coordinates
        crop_bounds: (left, top, right, bottom) pixel coordinates of crop
        original_img_size: [width, height] of original image
    
    Returns:
        new_bbox, new_point (both normalized to cropped image)
    """
    orig_width, orig_height = original_img_size
    left, top, right, bottom = crop_bounds
    crop_width = right - left
    crop_height = bottom - top
    
    # Convert original normalized coordinates to pixels
    bbox_x1_pixel = original_bbox[0] * orig_width
    bbox_y1_pixel = original_bbox[1] * orig_height
    bbox_x2_pixel = original_bbox[2] * orig_width
    bbox_y2_pixel = original_bbox[3] * orig_height
    
    point_x_pixel = original_point[0] * orig_width
    point_y_pixel = original_point[1] * orig_height
    
    # Adjust coordinates relative to crop
    new_bbox_x1 = max(0, bbox_x1_pixel - left)
    new_bbox_y1 = max(0, bbox_y1_pixel - top)
    new_bbox_x2 = min(crop_width, bbox_x2_pixel - left)
    new_bbox_y2 = min(crop_height, bbox_y2_pixel - top)
    
    new_point_x = max(0, min(crop_width, point_x_pixel - left))
    new_point_y = max(0, min(crop_height, point_y_pixel - top))
    
    # Normalize to cropped image size
    new_bbox = [
        new_bbox_x1 / crop_width,
        new_bbox_y1 / crop_height,
        new_bbox_x2 / crop_width,
        new_bbox_y2 / crop_height
    ]
    
    new_point = [
        new_point_x / crop_width,
        new_point_y / crop_height
    ]
    
    return new_bbox, new_point



def process_json_file(json_file_path, output_dir="output"):
    """
    Process the JSON file and create cropped images with recalculated coordinates.
    Preserves the original format with multiple elements per image.
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create output directories
    level1_dir = os.path.join(output_dir, "level1_crops")
    level2_dir = os.path.join(output_dir, "level2_crops")
    os.makedirs(level1_dir, exist_ok=True)
    os.makedirs(level2_dir, exist_ok=True)
    
    level1_data = []
    level2_data = []
    
    for item_idx, item in enumerate(data):
        img_name = item["img_url"]
        img_size = item["img_size"]
        elements = item["element"]
        
        img_url = 'data/data_dir/ShowUI-desktop/images/'+img_name
        
        # Check if image file exists
        if not os.path.exists(img_url):
            print(f"Warning: Image file {img_url} not found. Skipping.")
            continue
        
        # Generate base filename - keep original name for all crops
        original_filename = os.path.basename(img_name)
        original_name = os.path.splitext(original_filename)[0]
        original_ext = os.path.splitext(original_filename)[1] or '.png'
        
        # Process each element separately to create individual crops
        for elem_idx, element in enumerate(elements):
            instruction = element["instruction"]
            bbox = element["bbox"]
            point = element["point"]
            
            # Use original filename for all crops (they'll be in different directories)
            level1_filename = f"{original_name}{original_ext}"
            
            # Create separate directories for each element to avoid filename conflicts
            level1_elem_dir = os.path.join(level1_dir, f"elem_{elem_idx}")
            level2_elem_dir = os.path.join(level2_dir, f"elem_{elem_idx}")
            os.makedirs(level1_elem_dir, exist_ok=True)
            os.makedirs(level2_elem_dir, exist_ok=True)
            
            # Level 1 crop centered on this element's point
            cropped_img1, crop_bounds1 = crop_image_around_point(img_url, point)
            
            level1_path = os.path.join(level1_elem_dir, level1_filename)
            cropped_img1.save(level1_path)
            
            # Recalculate coordinates for level 1
            new_bbox1, new_point1 = recalculate_coordinates(bbox, point, crop_bounds1, img_size)
            
            # Create level 1 data entry
            level1_entry = {
                "img_url": f"level1_crops/elem_{elem_idx}/{level1_filename}",
                "img_size": list(cropped_img1.size),
                "element": [{
                    "instruction": instruction,
                    "bbox": new_bbox1,
                    "point": new_point1
                }]
            }
            level1_data.append(level1_entry)
            
            # Level 2 crop using the recalculated point from level 1
            cropped_img2, crop_bounds2 = crop_image_around_point(level1_path, new_point1)
            
            level2_path = os.path.join(level2_elem_dir, level1_filename)
            cropped_img2.save(level2_path)
            
            # Recalculate coordinates for level 2
            new_bbox2, new_point2 = recalculate_coordinates(new_bbox1, new_point1, crop_bounds2, cropped_img1.size)
            
            # Create level 2 data entry
            level2_entry = {
                "img_url": f"level2_crops/elem_{elem_idx}/{level1_filename}",
                "img_size": list(cropped_img2.size),
                "element": [{
                    "instruction": instruction,
                    "bbox": new_bbox2,
                    "point": new_point2
                }]
            }
            level2_data.append(level2_entry)
    
    # Save JSON files
    level1_json_path = os.path.join(output_dir, "level1_data.json")
    level2_json_path = os.path.join(output_dir, "level2_data.json")
    
    with open(level1_json_path, 'w') as f:
        json.dump(level1_data, f, indent=2)
    
    with open(level2_json_path, 'w') as f:
        json.dump(level2_data, f, indent=2)
    
    print(f"Processing complete!")
    print(f"Level 1 crops: {len(level1_data)} images in {level1_dir}")
    print(f"Level 2 crops: {len(level2_data)} images in {level2_dir}")
    print(f"JSON files saved: {level1_json_path}, {level2_json_path}")

def main():
    """
    Main function to run the script.
    Usage: python script.py
    
    Make sure your JSON file is named 'input_data.json' and in the same directory,
    or modify the file path below.
    """
    json_file_path = "data/data_dir/ShowUI-desktop/metadata/hf_train.json"  # Change this to your JSON file path
    output_directory = "cropped_images_all"  # Change this to your desired output directory
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found.")
        print("Please make sure the file exists or update the file path in the script.")
        return
    
    try:
        process_json_file(json_file_path, output_directory)
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()