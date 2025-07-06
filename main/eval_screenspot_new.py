import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ast
from typing import Dict, List, Any, Optional
import tempfile
import shutil
import logging
from tqdm import tqdm
import pickle
import torch.distributed as dist
from accelerate.utils import gather_object
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_json

class CroppedScreenSpotDataset(Dataset):
    """
    Dataset class for handling cropped images in the second round of inference.
    Loads cropped images and their corresponding metadata.
    """
    
    def __init__(self, cropped_dataset_info: List[Dict], processor, transform=None):
        """
        Args:
            cropped_dataset_info: List of dictionaries containing cropped image info
            processor: The model processor for tokenization and image processing
            transform: Optional image transforms
        """
        self.data = cropped_dataset_info
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load cropped image
        image_path = item['img_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            # Create a dummy image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Prepare instruction text
        instruction = item['instruction']
        
        # Process with the model processor
        processed = self.processor(
            text=instruction,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Prepare metadata
        meta_data = {
            'id': item['anno_id'],
            'img_url_abs': item['original_img_path'],  # Keep original path for reference
            'cropped_img_path': image_path,
            'task': instruction,
            'bbox': item['bbox'],
            'split': item['split'],
            'data_type': item['data_type'],
            'img_size': item['meta']['img_size'],
            'first_round_prediction': item['first_round_prediction']
        }
        
        return {
            'pixel_values': processed['pixel_values'].squeeze(0),
            'input_ids': processed['input_ids'].squeeze(0),
            'labels': processed['input_ids'].squeeze(0),  # For generation, labels = input_ids
            'meta_data': [meta_data]
        }


class TwoRoundDatasetManager:
    """
    Manager class for handling the two-round inference process with cropped images.
    """
    
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self.crop_temp_dir = None
        self.crop_metadata_dict = {}
        
    def create_cropped_dataset(self, first_round_outputs: List[Dict], crop_ratio: float = 0.5) -> List[Dict]:
        """
        Create cropped images and dataset info for the second round.
        
        Args:
            first_round_outputs: Results from the first round of inference
            crop_ratio: Ratio for cropping (0.5 = 1/2 width × 1/2 height = 1/4 area)
            
        Returns:
            List of dictionaries containing cropped dataset information
        """
        # Create temporary directory for cropped images
        self.crop_temp_dir = tempfile.mkdtemp(prefix="screenspot_crop_")
        logging.info(f"Created temporary directory for cropped images: {self.crop_temp_dir}")
        
        cropped_dataset_info = []
        
        for i, output in enumerate(tqdm(first_round_outputs, desc="Creating Cropped Images")):
            anno_id = output['anno_id']
            img_path = output['img_path']
            
            try:
                # Parse predicted point from first round
                pred_text = output['sentence']
                pred_point = ast.literal_eval(pred_text)
                
                # Crop image around predicted point
                cropped_image, crop_metadata = self.crop_image_around_point(
                    img_path, pred_point, crop_ratio=crop_ratio
                )
                
                # Save cropped image
                cropped_img_path = os.path.join(self.crop_temp_dir, f"cropped_{anno_id}.jpg")
                cropped_image.save(cropped_img_path)
                
                # Store metadata for coordinate conversion
                self.crop_metadata_dict[anno_id] = crop_metadata
                
                # Prepare info for second round dataset
                cropped_info = output.copy()
                cropped_info['img_path'] = cropped_img_path
                cropped_info['original_img_path'] = img_path
                cropped_info['first_round_prediction'] = pred_text
                cropped_dataset_info.append(cropped_info)
                
            except Exception as e:
                logging.warning(f"Error processing image {anno_id}: {e}")
                # Create a fallback entry for failed cases
                cropped_info = output.copy()
                cropped_info['img_path'] = img_path  # Use original image as fallback
                cropped_info['original_img_path'] = img_path
                cropped_info['first_round_prediction'] = output['sentence']
                cropped_dataset_info.append(cropped_info)
        
        # Save metadata for coordinate conversion
        metadata_path = os.path.join(self.crop_temp_dir, "crop_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.crop_metadata_dict, f, indent=2)
        
        logging.info(f"Created {len(cropped_dataset_info)} cropped images for second round")
        return cropped_dataset_info
    
    def crop_image_around_point(self, image_path: str, point_normalized: List[float], crop_ratio: float = 0.5):
        """
        Crop image around predicted point with specified ratio.
        
        Args:
            image_path: Path to original image
            point_normalized: Predicted point in normalized coordinates [x, y] in [0, 1]
            crop_ratio: Ratio of original image size for cropped image
            
        Returns:
            tuple: (cropped_image, crop_metadata)
        """
        image = Image.open(image_path)
        orig_width, orig_height = image.size
        
        # Calculate crop dimensions
        crop_width = int(orig_width * crop_ratio)
        crop_height = int(orig_height * crop_ratio)
        
        # Convert normalized point to pixel coordinates
        center_x = int(point_normalized[0] * orig_width)
        center_y = int(point_normalized[1] * orig_height)
        
        # Calculate crop boundaries
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        right = min(orig_width, left + crop_width)
        bottom = min(orig_height, top + crop_height)
        
        # Adjust if crop goes beyond image boundaries
        if right - left < crop_width:
            if left == 0:
                right = min(orig_width, crop_width)
            else:
                left = max(0, orig_width - crop_width)
        
        if bottom - top < crop_height:
            if top == 0:
                bottom = min(orig_height, crop_height)
            else:
                top = max(0, orig_height - crop_height)
        
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        
        # Store metadata for coordinate conversion
        crop_metadata = {
            'orig_size': (orig_width, orig_height),
            'crop_size': (right - left, bottom - top),
            'crop_bounds': (left, top, right, bottom),
            'center_point_pixel': (center_x, center_y),
            'center_point_normalized': point_normalized
        }
        
        return cropped_image, crop_metadata
    
    def convert_cropped_point_to_original(self, cropped_point_normalized: List[float], crop_metadata: Dict) -> List[float]:
        """
        Convert point from cropped image coordinates back to original image coordinates.
        
        Args:
            cropped_point_normalized: Point in cropped image normalized coordinates [x, y]
            crop_metadata: Metadata from crop_image_around_point function
            
        Returns:
            Point in original image normalized coordinates [x, y]
        """
        crop_bounds = crop_metadata['crop_bounds']
        orig_size = crop_metadata['orig_size']
        crop_size = crop_metadata['crop_size']
        
        # Convert from cropped normalized to cropped pixel coordinates
        cropped_x_pixel = cropped_point_normalized[0] * crop_size[0]
        cropped_y_pixel = cropped_point_normalized[1] * crop_size[1]
        
        # Convert to original image pixel coordinates
        orig_x_pixel = crop_bounds[0] + cropped_x_pixel
        orig_y_pixel = crop_bounds[1] + cropped_y_pixel
        
        # Convert back to normalized coordinates
        original_point_normalized = [
            orig_x_pixel / orig_size[0],
            orig_y_pixel / orig_size[1]
        ]
        
        return original_point_normalized
    
    def create_second_round_dataloader(self, cropped_dataset_info: List[Dict], batch_size: int = 1, 
                                     num_workers: int = 0, shuffle: bool = False) -> DataLoader:
        """
        Create a DataLoader for the second round of inference.
        
        Args:
            cropped_dataset_info: List of cropped dataset information
            batch_size: Batch size for the DataLoader
            num_workers: Number of worker processes
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the second round
        """
        dataset = CroppedScreenSpotDataset(cropped_dataset_info, self.processor)
        
        # Custom collate function to handle the specific format
        def collate_fn(batch):
            # Assuming batch size of 1 for simplicity, but can be extended
            if len(batch) == 1:
                return batch[0]
            else:
                # Handle multiple samples in batch
                pixel_values = torch.stack([item['pixel_values'] for item in batch])
                input_ids = torch.stack([item['input_ids'] for item in batch])
                labels = torch.stack([item['labels'] for item in batch])
                meta_data = [item['meta_data'][0] for item in batch]
                
                return {
                    'pixel_values': pixel_values,
                    'input_ids': input_ids,
                    'labels': labels,
                    'meta_data': meta_data
                }
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return dataloader
    
    def cleanup(self):
        """Clean up temporary directory."""
        if self.crop_temp_dir and os.path.exists(self.crop_temp_dir):
            logging.info(f"Cleaning up temporary directory: {self.crop_temp_dir}")
            shutil.rmtree(self.crop_temp_dir)
            self.crop_temp_dir = None
            
def safe_gather_object_chunked(obj: Any, chunk_size: int = 1000000) -> List[Any]:
    """
    Even safer version that processes in chunks to avoid large tensors
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    logging.info(f"Rank {rank}: Starting chunked gather")
    
    try:
        # Only gather from rank 0 to avoid large collective operations
        if rank == 0:
            # Collect from all ranks using point-to-point communication
            gathered_objects = [obj]  # Include rank 0's object
            
            for src_rank in range(1, world_size):
                try:
                    # Receive size first
                    size_tensor = torch.zeros(1, dtype=torch.long).cuda()
                    dist.recv(size_tensor, src=src_rank)
                    obj_size = size_tensor.item()
                    
                    # Receive object data
                    obj_tensor = torch.zeros(obj_size, dtype=torch.uint8).cuda()
                    dist.recv(obj_tensor, src=src_rank)
                    
                    # Deserialize
                    obj_bytes = obj_tensor.cpu().numpy().tobytes()
                    received_obj = pickle.loads(obj_bytes)
                    gathered_objects.append(received_obj)
                    
                except Exception as e:
                    logging.error(f"Failed to receive from rank {src_rank}: {e}")
                    gathered_objects.append(None)  # Placeholder
            
            return gathered_objects
            
        else:
            # Send to rank 0
            obj_bytes = pickle.dumps(obj)
            obj_size = len(obj_bytes)
            
            # Send size first
            size_tensor = torch.tensor([obj_size], dtype=torch.long).cuda()
            dist.send(size_tensor, dst=0)
            
            # Send object data
            obj_tensor = torch.frombuffer(obj_bytes, dtype=torch.uint8).cuda()
            dist.send(obj_tensor, dst=0)
            
            return []  # Non-root ranks don't need the gathered result
            
    except Exception as e:
        logging.error(f"Rank {rank}: chunked gather failed: {e}")
        return [obj] if rank == 0 else []

def perform_inference_round(val_loader, model_engine, processor, local_rank, args, round_name=""):
    """
    Robust version of perform_inference_round with enhanced error handling
    """
    model_engine.eval()
    
    generated_texts_unique = []
    outputs_unique = []
    successful_inferences = 0
    failed_inferences = 0
    
    logging.info(f"Starting {round_name} inference round...")
    
    for i, input_dict in enumerate(tqdm(val_loader, desc=f"{round_name} Inference")):
        try:
            torch.cuda.empty_cache()
            
            # Move input to GPU
            input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

            # Handle different precision modes
            if hasattr(args, 'precision'):
                if args.precision == "fp16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].half()
                elif args.precision == "bf16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
                else:
                    input_dict["pixel_values"] = input_dict["pixel_values"].float()
            else:
                input_dict["pixel_values"] = input_dict["pixel_values"].float()

            with torch.no_grad():
                # Prepare forward dictionary
                forward_dict = {
                    "pixel_values": input_dict["pixel_values"],
                    "input_ids": input_dict["input_ids"],
                }
                
                # Add labels if they exist (for some model architectures)
                if "labels" in input_dict:
                    forward_dict["labels"] = input_dict["labels"]

                # Add optional inputs if they exist
                optional_keys = ["image_sizes", "patch_assign", "patch_assign_len", "patch_pos", "select_mask"]
                for key in optional_keys:
                    if key in input_dict:
                        if key == "image_sizes":
                            forward_dict["image_grid_thw"] = input_dict[key]
                        else:
                            forward_dict[key] = input_dict[key]
                
                # Generate predictions
                generate_ids = model_engine.generate(
                    **forward_dict, 
                    max_new_tokens=128, 
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else processor.tokenizer.eos_token_id,
                    temperature=1.0,
                    top_p=1.0,
                )
                
                # Remove input tokens from generated sequence
                generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
                
                # Decode generated tokens to text
                generated_texts = processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
                successful_inferences += 1
                
        except Exception as e:
            logging.warning(f"Error during {round_name} inference on sample {i}: {e}")
            generated_texts = "[0.5, 0.5]"  # Default fallback prediction
            failed_inferences += 1

        # Get metadata
        meta = input_dict['meta_data'][0] if 'meta_data' in input_dict else {}
        
        # Log progress for every 10th sample
        if i % 10 == 0:
            logging.info(f"{round_name} - Sample {i}: generated_texts: {generated_texts}")
            if 'bbox' in meta:
                logging.info(f"{round_name} - Sample {i}: ground truth bbox: {meta['bbox']}")

        # Create output dictionary
        outputs = {
            "split": meta.get('split', 'unknown'), 
            "data_type": meta.get('data_type', 'unknown'),
            "anno_id": meta.get('id', f'sample_{i}'), 
            "img_path": meta.get('img_url_abs', ''), 
            "instruction": meta.get('task', ''), 
            "sentence": generated_texts,
            "bbox": meta.get('bbox', []), 
            "meta": meta
        }

        generated_texts_unique.append(generated_texts)
        outputs_unique.append(outputs)
    
    # Log statistics
    logging.info(f"{round_name} inference completed:")
    logging.info(f"  - Successful inferences: {successful_inferences}")
    logging.info(f"  - Failed inferences: {failed_inferences}")
    logging.info(f"  - Total samples: {len(generated_texts_unique)}")
    
    # Gather results from all processes in distributed setting
    generated_texts_unique = safe_gather_object_chunked(generated_texts_unique)
    outputs_unique = safe_gather_object_chunked(outputs_unique)
    
    return generated_texts_unique, outputs_unique


# Modified version of the main validation function
def validate_screenspot(val_loader, model_engine, processor, epoch, global_step, writer, args, media=True):
    """
    Enhanced validation function with proper two-round inference using cropped images.
    """
    model_engine.eval()
    
    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize dataset manager
    dataset_manager = TwoRoundDatasetManager(processor, args)
    
    try:
        # First round of inference
        logging.info("Starting first round of inference...")
        first_round_texts, first_round_outputs = perform_inference_round(
            val_loader, model_engine, processor, local_rank, args, "First Round"
        )
        
        if global_rank == 0:
            # Create cropped dataset for second round
            cropped_dataset_info = dataset_manager.create_cropped_dataset(
                first_round_outputs, crop_ratio=0.5
            )
            
            # Create second round data loader
            second_round_loader = dataset_manager.create_second_round_dataloader(
                cropped_dataset_info, batch_size=1, num_workers=0, shuffle=False
            )
            
            logging.info(f"Created second round dataloader with {len(cropped_dataset_info)} samples")
        else:
            second_round_loader = None
        
        # Synchronize all processes
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
        
        # Second round of inference with cropped images
        if global_rank == 0:
            logging.info("Starting second round of inference with cropped images...")
            second_round_texts, second_round_outputs = perform_inference_round(
                second_round_loader, model_engine, processor, local_rank, args, "Second Round"
            )
        else:
            second_round_texts, second_round_outputs = [], []
        
        # Broadcast results to all processes if needed
        if world_size > 1:
            # Gather results from rank 0 to all processes
            from accelerate.utils import gather_object
            second_round_texts = safe_gather_object_chunked(second_round_texts)
            second_round_outputs = safe_gather_object_chunked(second_round_outputs)
        
        if global_rank == 0:
            # Process results and convert coordinates back to original image space
            results = process_second_round_results(
                second_round_texts, second_round_outputs, dataset_manager, args
            )
            
            # Calculate and log metrics
            eval_dict = calculate_metrics_and_log(results, epoch, global_step, writer, args)
            
            # Create visualizations
            if media:
                create_visualizations(results, global_step, args)
            
            # Save results
            save_results(results, eval_dict, epoch, args)
            
            # Get final metric
            score_all = [value for split in eval_dict.values() for value in split.values()]
            metric = sum(score_all) / len(score_all)
            
        else:
            metric = 0.0
            
    finally:
        # Clean up
        if global_rank == 0:
            dataset_manager.cleanup()
    
    # Synchronize all processes before returning
    if world_size > 1:
        dist.barrier()
    
    # Broadcast metric to all processes
    if world_size > 1:
        metric_tensor = torch.tensor([metric], dtype=torch.float32).to(f'cuda:{local_rank}')
        dist.broadcast(metric_tensor, src=0)
        metric = metric_tensor.item()
    
    return metric


def process_second_round_results(second_round_texts, second_round_outputs, dataset_manager, args):
    """Process second round results and convert coordinates back to original image space."""
    results = {}
    
    logging.info("Processing second round results and converting coordinates...")
    
    for pred_text, output in tqdm(zip(second_round_texts, second_round_outputs), desc="Processing Second Round"):
        anno_id = output['anno_id']
        split_i = output['split']
        type_i = output['data_type']
        
        if split_i not in results:
            results[split_i] = {}
        if type_i not in results[split_i]:
            results[split_i][type_i] = []
        
        step_result = output.copy()
        img_size = output['meta']['img_size']
        
        # Import the get_bbox and pointinbbox functions from your original code
        from eval_screenspot import get_bbox, pointinbbox  
        
        gt_bbox = get_bbox(output['bbox'], img_size, args.xy_int)
        step_result['gt_bbox'] = gt_bbox
        
        try:
            # Parse second round prediction
            second_round_point = ast.literal_eval(pred_text)
            
            # Convert coordinates back to original image space
            if anno_id in dataset_manager.crop_metadata_dict:
                # Convert from cropped coordinates to original coordinates
                original_point = dataset_manager.convert_cropped_point_to_original(
                    second_round_point, dataset_manager.crop_metadata_dict[anno_id]
                )
                step_result['pred_point'] = original_point
                step_result['second_round_point_cropped'] = second_round_point
            else:
                # Fallback to second round point if no crop metadata
                step_result['pred_point'] = second_round_point
                step_result['second_round_point_cropped'] = second_round_point
            
            # Calculate accuracy
            if pointinbbox(step_result['pred_point'], gt_bbox):
                step_result["acc"] = 1
            else:
                step_result["acc"] = 0
                
        except Exception as e:
            logging.warning(f"Error processing second round result for {anno_id}: {e}")
            step_result["acc"] = 0
        
        results[split_i][type_i].append(step_result)
    
    return results


def calculate_metrics_and_log(results, epoch, global_step, writer, args):
    """Calculate metrics and log them."""
    eval_dict = {}

    from eval_screenspot import calculate_screenspot_metrics 
    
    for split in results.keys():
        logging.info("==="*10)
        logging.info(f"{split}")
        logging.info("==="*10)
        eval_dict[split] = calculate_screenspot_metrics(results[split])
    
    # Log metrics
    if not args.debug:
        import wandb
        for split in eval_dict.keys():
            for key, value in eval_dict[split].items():
                if isinstance(value, list):
                    continue
                writer.add_scalar(f"metrics/screenspot_two_round/{split}/{key}", value, epoch)
                wandb.log({f"metrics/screenspot_two_round/{split}/{key}": value}, step=global_step)
    
    return eval_dict


def create_visualizations(results, global_step, args):
    """Create visualization examples."""
    import wandb
    import random
    from eval_screenspot import draw_point_bbox  
    
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
            
            images = wandb.Image(img_array, caption=f"TwoRound/{split}/{type}/{img_anno}_{img_inst}")
            images_list.append(images)
    
    wandb.log({"two_round_examples": images_list}, step=global_step)


def save_results(results, eval_dict, epoch, args):
    """Save results to JSON files."""
    from eval_screenspot import save_json  
    
    save_json(results, os.path.join(args.tmp_dir, f'screenspot_two_round_epo{epoch}_tmp_dict.json'))
    save_json(eval_dict, os.path.join(args.tmp_dir, f'screenspot_two_round_epo{epoch}_res_dict.json'))