import os
import re
import ast
import sys
import pdb
import json
import torch
import wandb
import random
import numpy as np
import tempfile
import gc
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.distributed as dist
from accelerate.utils import gather_object
from torchvision.transforms.functional import crop
from functools import partial
from data.dataset import collate_fn
print("CWD:", os.getcwd())
print("Files in CWD:", os.listdir('.'))
print("data/data_utils.py exists:", os.path.isfile('data/data_utils.py'))

sys.path.append(os.getcwd())

from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_json

import logging
logging.basicConfig(level=logging.INFO)

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def get_bbox(bbox, img_size, xy_int):
    x1, y1, w, h = bbox
    weight, height = img_size

    # x1y1wh to x1y1x2y2
    bbox = [x1, y1, x1 + w, y1 + h]

    # normalisation
    bbox = [bbox[0] / weight, bbox[1] / height, 
            bbox[2] / weight, bbox[3] / height]
    if xy_int:
        bbox = [int(item * 1000) for item in bbox]
    return bbox

def pointinbbox(pred_point, gt_bbox):
    # pred_point: [x, y] in [0, 1]
    # gt_bbox: [x1, y1, x2, y2] in [0, 1]
    if (gt_bbox[0] <= pred_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_point[1] <= gt_bbox[3]):
        return True
    else:
        return False

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0] * width, point[1] * height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    image_draw = np.array(image)
    return image_draw

def calculate_screenspot_metrics(results):
    metrics = {}
    for type in results:
        num_step = 0
        num_success = 0

        for step in results[type]:
            num_step += 1
            num_success += step["acc"]

        metrics[f"{type} Success Rate"] = num_success / num_step

    for key, value in metrics.items():
        logging.info(f"[{key}]: {value}")
    return metrics

def crop_image_around_point(image_path, pred_point, crop_ratio=0.5, cleanup_original=False):
    """
    Crop image to 1/4 original size (1/2 width and height) centered around predicted point.
    
    Args:
        image_path: Path to original image
        pred_point: Predicted point in normalized coordinates [0, 1]
        crop_ratio: Ratio for cropping (0.5 means 1/2 of original dimensions)
        cleanup_original: Whether to delete the original image file after cropping
    
    Returns:
        cropped_image: PIL Image object
        crop_info: Dictionary with cropping information for coordinate conversion
    """
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    
    # Calculate crop dimensions
    crop_width = int(original_width * crop_ratio)
    crop_height = int(original_height * crop_ratio)
    
    # Convert normalized point to pixel coordinates
    pred_x = pred_point[0] * original_width
    pred_y = pred_point[1] * original_height
    
    # Calculate crop bounds, ensuring they stay within image boundaries
    left = max(0, int(pred_x - crop_width // 2))
    top = max(0, int(pred_y - crop_height // 2))
    right = min(original_width, left + crop_width)
    bottom = min(original_height, top + crop_height)
    
    # Adjust left/top if right/bottom hit boundaries
    if right == original_width:
        left = max(0, right - crop_width)
    if bottom == original_height:
        top = max(0, bottom - crop_height)
    
    # Crop the image
    cropped_image = crop(original_image, top, left, crop_height, crop_width)
    
    # Resize back to original dimensions to maintain model compatibility
    cropped_image = cropped_image.resize((original_width, original_height), Image.LANCZOS)
    
    # Clean up original image from memory
    original_image.close()
    del original_image
    
    # Optionally delete the original file
    if cleanup_original and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"Deleted original image: {image_path}")
        except OSError as e:
            print(f"Warning: Could not delete {image_path}: {e}")
    
    crop_info = {
        'left': left,
        'top': top,
        'right': right,
        'bottom': bottom,
        'crop_width': crop_width,
        'crop_height': crop_height,
        'original_width': original_width,
        'original_height': original_height
    }
    
    return cropped_image, crop_info

def convert_cropped_coords_to_original(cropped_point, crop_info):
    """
    Convert coordinates from cropped image back to original image coordinates.
    
    Args:
        cropped_point: Point in normalized coordinates [0, 1] relative to cropped image
        crop_info: Dictionary with cropping information
    
    Returns:
        original_point: Point in normalized coordinates [0, 1] relative to original image
    """
    # Map the prediction from crop coordinates to original image coordinates
    original_x = (crop_info['left'] + cropped_point[0] * crop_info['crop_width']) / crop_info['original_width']
    original_y = (crop_info['top'] + cropped_point[1] * crop_info['crop_height']) / crop_info['original_height']
    
    return [original_x, original_y]

def create_cropped_input_dict(cropped_image, task_text, processor, original_input_dict, args, local_rank):
    """
    Create input dictionary for cropped image inference.
    """
    # Process the cropped image
    processed_inputs = processor(
        text=task_text,
        images=cropped_image,
        return_tensors="pt"
    )
    
    cropped_input_dict = {
        "pixel_values": processed_inputs["pixel_values"],
        "input_ids": processed_inputs["input_ids"]
    }
    
    # Copy necessary fields from original input
    fields_to_copy = ["image_sizes", "patch_assign", "patch_assign_len", "patch_pos", "select_mask", "labels"]
    for field in fields_to_copy:
        if field in original_input_dict:
            cropped_input_dict[field] = original_input_dict[field]
    
    # Move to device and apply precision
    cropped_input_dict = dict_to_cuda(cropped_input_dict, device=f'cuda:{local_rank}')
    
    if args.precision == "fp16":
        cropped_input_dict["pixel_values"] = cropped_input_dict["pixel_values"].half()
    elif args.precision == "bf16":
        cropped_input_dict["pixel_values"] = cropped_input_dict["pixel_values"].bfloat16()
    else:
        cropped_input_dict["pixel_values"] = cropped_input_dict["pixel_values"].float()
    
    return cropped_input_dict

def create_forward_dict(input_dict):
    """
    Create forward dictionary for model inference.
    """
    forward_dict = {
        "pixel_values": input_dict["pixel_values"],
        "input_ids": input_dict["input_ids"],
    }
    
    # Add optional fields
    optional_fields = [
        ("image_sizes", "image_grid_thw"),
        ("patch_assign", "patch_assign"),
        ("patch_assign_len", "patch_assign_len"),
        ("patch_pos", "patch_pos"),
        ("select_mask", "select_mask"),
        ("labels", "labels")
    ]
    
    for input_key, forward_key in optional_fields:
        if input_key in input_dict:
            forward_dict[forward_key] = input_dict[input_key]
    
    return forward_dict

def run_inference(model_engine, forward_dict, processor, max_new_tokens=128):
    """
    Run model inference and return generated text.
    """
    with torch.no_grad():
        generate_ids = model_engine.generate(
            **forward_dict,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        
        generate_ids = generate_ids[:, forward_dict['input_ids'].shape[1]:]
        generated_text = processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
    return generated_text



class CroppedImageDataset:
    """Dataset for cropped images in second round inference with memory cleanup."""
    
    def __init__(self, original_samples, first_predictions, crop_infos, cleanup_temp_files=True):
        self.samples = []
        self.temp_files = []  # Track temporary files for cleanup
        self.cleanup_temp_files = cleanup_temp_files
        
        for sample, pred_point, crop_info in zip(original_samples, first_predictions, crop_infos):
            cropped_path = self._create_cropped_image(sample, pred_point)
            self.samples.append({
                'original_sample': sample,
                'first_pred_point': pred_point,
                'crop_info': crop_info,
                'cropped_image_path': cropped_path
            })
            self.temp_files.append(cropped_path)
    
    def _create_cropped_image(self, sample, pred_point):
        """Create and save cropped image, return path."""
        meta = sample['meta_data'][0]
        img_path = meta['img_url_abs']
        
        # Create cropped image (don't delete original here as it might be needed elsewhere)
        cropped_image, _ = crop_image_around_point(img_path, pred_point, cleanup_original=False)
        
        # Save cropped image temporarily
        temp_dir = tempfile.mkdtemp()
        cropped_path = os.path.join(temp_dir, f"cropped_{meta['id']}.jpg")
        cropped_image.save(cropped_path)
        
        # Clean up cropped image from memory immediately after saving
        cropped_image.close()
        del cropped_image
        gc.collect()  # Force garbage collection
        
        return cropped_path
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_data = self.samples[idx]
        original_sample = sample_data['original_sample']
        meta = original_sample['meta_data'][0]
        
        # Create new sample with cropped image path
        cropped_sample = {
            'image_path': sample_data['cropped_image_path'],
            'task': meta.get('task', ''),
            'meta_data': [{
                **meta,
                'img_url_abs': sample_data['cropped_image_path'],  # Use cropped image path
                'crop_info': sample_data['crop_info'],
                'first_pred_point': sample_data['first_pred_point']
            }]
        }
        
        return cropped_sample
    
    def cleanup_temp_files(self):
        """Clean up all temporary cropped image files."""
        if not self.cleanup_temp_files:
            return
            
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")
                # Also try to remove the temporary directory if empty
                temp_dir = os.path.dirname(temp_file)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except OSError as e:
                print(f"Warning: Could not clean up {temp_file}: {e}")
        
        self.temp_files.clear()
    
    def __del__(self):
        """Ensure cleanup happens when dataset is destroyed."""
        self.cleanup_temp_files()

@torch.no_grad()
def validate_screenspot(val_loader, model_engine, processor, epoch, global_step, writer, args, media=True):
    model_engine.eval()

    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    metric = 0
    
    # First round: collect all samples and predictions
    first_round_samples = []
    first_round_predictions = []
    first_round_crop_infos = []
    
    print("Starting first round inference...")
    for i, input_dict in enumerate(tqdm(val_loader)):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        meta = input_dict['meta_data'][0]
        
        try:
            # First round inference on original image
            forward_dict = create_forward_dict(input_dict)
            generated_text = run_inference(model_engine, forward_dict, processor)
            
            print(f"Sample {i} - First round generated_texts: {generated_text}")
            
            # Parse the first prediction
            try:
                first_pred_point = ast.literal_eval(generated_text)
                print(f"Sample {i} - First prediction point: {first_pred_point}")
            except Exception as e:
                print(f"Sample {i} - Error parsing first prediction: {e}")
                first_pred_point = [0.5, 0.5]
            
            # Calculate crop info for this prediction
            img_path = meta['img_url_abs']
            _, crop_info = crop_image_around_point(img_path, first_pred_point)
            
            # Store data for second round
            first_round_samples.append(input_dict)
            first_round_predictions.append(first_pred_point)
            first_round_crop_infos.append(crop_info)
            
            # Clean up GPU memory after each sample
            del forward_dict
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Sample {i} - Error in first round inference: {e}")
            import traceback
            traceback.print_exc()
            first_round_samples.append(input_dict)
            first_round_predictions.append([0.5, 0.5])
            first_round_crop_infos.append(None)

    # Force garbage collection after first round
    gc.collect()
    torch.cuda.empty_cache()

    # Gather first round results from all processes
    first_round_samples = gather_object(first_round_samples)
    first_round_predictions = gather_object(first_round_predictions) 
    first_round_crop_infos = gather_object(first_round_crop_infos)
    
    # Second round: create new dataset and loader for cropped images
    if global_rank == 0:
        print("Creating cropped image dataset for second round...")
        
        # Filter out samples with None crop_info
        valid_samples = []
        valid_predictions = []
        valid_crop_infos = []
        
        for sample, pred, crop_info in zip(first_round_samples, first_round_predictions, first_round_crop_infos):
            if crop_info is not None:
                valid_samples.append(sample)
                valid_predictions.append(pred)
                valid_crop_infos.append(crop_info)
        
        # Create cropped dataset with cleanup enabled
        cropped_dataset = CroppedImageDataset(valid_samples, valid_predictions, valid_crop_infos, cleanup_temp_files=True)
        
        try:
            # Create new data loader for cropped images
            cropped_sampler = torch.utils.data.distributed.DistributedSampler(
                cropped_dataset, shuffle=False, drop_last=False
            ) if args.distributed else None
            
            cropped_val_loader = torch.utils.data.DataLoader(
                cropped_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=cropped_sampler,
                collate_fn=partial(collate_fn, processor=processor),
            )
            
            print(f"Created cropped dataset with {len(cropped_dataset)} samples")
            
            # Second round inference
            print("Starting second round inference on cropped images...")
            for i, cropped_input_dict in enumerate(tqdm(cropped_val_loader)):
                torch.cuda.empty_cache()
                
                # Move to device and apply precision (same as original)
                cropped_input_dict = dict_to_cuda(cropped_input_dict, device=f'cuda:{local_rank}')

                if args.precision == "fp16":
                    cropped_input_dict["pixel_values"] = cropped_input_dict["pixel_values"].half()
                elif args.precision == "bf16":
                    cropped_input_dict["pixel_values"] = cropped_input_dict["pixel_values"].bfloat16()
                else:
                    cropped_input_dict["pixel_values"] = cropped_input_dict["pixel_values"].float()

                meta = cropped_input_dict['meta_data'][0]
                crop_info = meta['crop_info']
                first_pred_point = meta['first_pred_point']
                
                try:
                    # Second round inference on cropped image
                    cropped_forward_dict = create_forward_dict(cropped_input_dict)
                    cropped_generated_text = run_inference(model_engine, cropped_forward_dict, processor)
                    
                    print(f"Sample {i} - Second round generated_texts: {cropped_generated_text}")
                    
                    # Parse cropped prediction and convert back to original coordinates
                    try:
                        cropped_pred_point = ast.literal_eval(cropped_generated_text)
                        final_pred_point = convert_cropped_coords_to_original(cropped_pred_point, crop_info)
                        
                        print(f"Sample {i} - Cropped prediction: {cropped_pred_point}")
                        print(f"Sample {i} - Final prediction (original coords): {final_pred_point}")
                        
                        pred_point = final_pred_point
                        generated_text = str(final_pred_point)
                        
                    except Exception as e:
                        print(f"Sample {i} - Error parsing/converting cropped prediction: {e}")
                        pred_point = first_pred_point
                        generated_text = str(first_pred_point)
                        
                    # Clean up GPU memory after each sample
                    del cropped_forward_dict
                    torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Sample {i} - Error in second round inference: {e}")
                    import traceback
                    traceback.print_exc()
                    pred_point = first_pred_point
                    generated_text = str(first_pred_point)

                # Create output dictionary (using original metadata)
                original_meta = {k: v for k, v in meta.items() if k not in ['crop_info', 'first_pred_point']}
                outputs = {
                    "split": original_meta['split'], 
                    'data_type': original_meta['data_type'],
                    "anno_id": original_meta['id'], 
                    "img_path": original_meta.get('img_url_abs_original', original_meta['img_url_abs']), 
                    "instruction": original_meta['task'], 
                    "sentence": generated_text,
                    "bbox": original_meta['bbox'], 
                    "meta": original_meta,
                    'pred_point': pred_point
                }

                generated_texts_unique.append(generated_text)
                answers_unique.append(original_meta['bbox'])
                outputs_unique.append(outputs)
                
        finally:
            # Always clean up temporary files, even if an error occurs
            print("Cleaning up temporary cropped image files...")
            cropped_dataset.cleanup_temp_files()
            del cropped_dataset
            gc.collect()
    
    else:
        # Non-master processes just wait
        generated_texts_unique = []
        answers_unique = []
        outputs_unique = []

    # Rest of the function remains the same...
    # Gather final results from all processes  
    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)
    outputs_unique = gather_object(outputs_unique)

    if global_rank == 0:
        results = {}
        for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
            anno_id = output_i['anno_id']
            split_i = output_i['split']
            if split_i not in results:
                results[split_i] = {}

            type_i = output_i['data_type']
            if type_i not in results[split_i]:
                results[split_i][type_i] = []

            step_result = output_i.copy()

            img_size = output_i['meta']['img_size']
            gt_bbox = get_bbox(ans_i, img_size, args.xy_int)
            step_result['gt_bbox'] = gt_bbox

            try:
                pred_point = output_i['pred_point']
                
                if pointinbbox(pred_point, gt_bbox):
                    step_result["acc"] = 1
                else:
                    step_result["acc"] = 0
                    
            except Exception as e:
                print(e)
                print(f"Error evaluating {anno_id}'s prediction: {pred_i}")
                step_result["acc"] = 0

            results[split_i][type_i].append(step_result)

        # Calculate metrics
        eval_dict = {}
        for split in results.keys():
            logging.info("==="*10)
            logging.info(f"{split}")
            logging.info("==="*10)
            eval_dict[split] = calculate_screenspot_metrics(results[split])

        # Log metrics
        if not args.debug:
            for split in eval_dict.keys():
                for key, value in eval_dict[split].items():
                    if isinstance(value, list):
                        continue
                    writer.add_scalar(f"metrics/screenspot/{split}/{key}", value, epoch)
                    wandb.log({f"metrics/screenspot/{split}/{key}": value}, step=global_step)

        # Calculate average metric
        score_all = [value for split in eval_dict.values() for value in split.values()]
        metric = sum(score_all) / len(score_all)
        eval_dict['Avg Success Rate'] = metric
        writer.add_scalar("metrics/screenspot/Avg Success Rate", metric, epoch)
        wandb.log({"metrics/screenspot/Avg Success Rate": metric}, step=global_step)

        # Log visual examples
        if media:
            images_list = []
            for split in results.keys():
                for type in results[split].keys():
                    sample = random.choice(results[split][type])
                    img_anno = sample['anno_id']
                    img_url = sample['img_path']
                    img_inst = sample['instruction']
                    gt_bbox = sample['gt_bbox']
                    if 'pred_point' in sample:
                        pred_point = sample['pred_point']
                        img_array = draw_point_bbox(img_url, pred_point, gt_bbox, radius=5, line=3)
                    else:
                        img_array = draw_point_bbox(img_url, None, gt_bbox)
                    images = wandb.Image(img_array, caption=f"{split}/{type}/{img_anno}_{img_inst}")
                    images_list.append(images)
            wandb.log({"examples": images_list}, step=global_step)
 
        save_json(results, os.path.join(args.tmp_dir, f'screenspot_epo{epoch}_tmp_dict.json'))
        save_json(eval_dict, os.path.join(args.tmp_dir, f'screenspot_epo{epoch}_res_dict.json'))

    metric = broadcast_value(metric, src=0, local_rank=local_rank)
    return metric