import argparse
import os
import datetime
import numpy as np
import cv2
from tqdm import tqdm
import py_sod_metrics


def parse_args():
    parser = argparse.ArgumentParser("Citation Format Evaluation Tool")
    parser.add_argument("--pred_path", type=str, default="results/RDVS_TEST",
                        help="path to the predicted masks (citation format)")
    parser.add_argument("--gt_path", type=str,  default="/data/RDVS/test",
                        help="path to the ground truth masks")
    parser.add_argument("--dataset", type=str, default="rdvs",
                        help="dataset name for output logging")
    parser.add_argument("--output_file", type=str, default="eval_results.txt",
                        help="output file to save evaluation results")
    return parser.parse_args()

def load_citation_format_predictions(pred_path):
    """
    Load predictions saved in citation format
    Expected structure: pred_path/video_name/frame_name.png
    """
    predictions = {}
    
    if not os.path.exists(pred_path):
        raise ValueError(f"Prediction path does not exist: {pred_path}")
    
    for video_dir in os.listdir(pred_path):
        video_path = os.path.join(pred_path, video_dir)
        if not os.path.isdir(video_path):
            continue
            
        predictions[video_dir] = {}
        for frame_file in os.listdir(video_path):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_name = os.path.splitext(frame_file)[0]
                frame_path = os.path.join(video_path, frame_file)
                predictions[video_dir][frame_name] = frame_path
                
    return predictions

def load_test_list(dataset, gt_path):
    """
    Load test clip list for specific datasets
    """
    if dataset == "dvisal":
        list_file = os.path.join(gt_path, "test_all.txt")
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                test_clips = [line.strip() for line in f.readlines() if line.strip()]
            return test_clips
        else:
            print(f"Warning: test_all.txt not found at {list_file}, using all clips")
            return None
    else:
        return None  # For other datasets, use all clips

def load_ground_truth(gt_path, dataset):
    """
    Load ground truth masks
    Expected structure: gt_path/video_name/gt_folder/frame_name.png
    """
    ground_truths = {}
    
    if not os.path.exists(gt_path):
        raise ValueError(f"Ground truth path does not exist: {gt_path}")
    
    # Get test list for specific datasets
    test_clips = load_test_list(dataset, gt_path)
    if test_clips:
        print(f"Loaded {len(test_clips)} test clips from list")
    
    for video_dir in os.listdir(gt_path):
        # For DViSal, only process clips in test_all.txt
        if test_clips is not None and video_dir not in test_clips:
            continue
            
        if dataset=="dvisal":
            gt_name="GT"
        elif dataset=="rdvs":
            gt_name="ground-truth"
        elif dataset=="vidsod":
            gt_name="gt"
        else:
            gt_name="GT"  # default
            
        video_path = os.path.join(gt_path, video_dir, gt_name)
        
        if not os.path.isdir(video_path):
            continue
            
        ground_truths[video_dir] = {}
        for frame_file in os.listdir(video_path):
            if frame_file.endswith('.png'):
                frame_name = frame_file[:-4]  # remove .png extension
                frame_path = os.path.join(video_path, frame_file)
                ground_truths[video_dir][frame_name] = frame_path
                
    return ground_truths

def evaluate_predictions(predictions, ground_truths):
    """
    Evaluate predictions using py_sod_metrics
    Only evaluates samples that exist in ground truth (test set)
    """
    # Initialize metrics (same as in test.py)
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    
    FM = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
            "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
            "rec": py_sod_metrics.RecallHandler(**sample_gray),
            "fpr": py_sod_metrics.FPRHandler(**sample_gray),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
            "dice": py_sod_metrics.DICEHandler(**sample_gray),
            "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
            "ber": py_sod_metrics.BERHandler(**sample_gray),
            "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
            "kappa": py_sod_metrics.KappaHandler(**sample_gray),
        }
    )
    
    total_samples = 0
    matched_samples = 0
    
    # Process each video based on ground truth (to ensure only test samples are evaluated)
    for video_name in tqdm(ground_truths.keys(), desc="Evaluating videos"):
        if video_name not in predictions:
            print(f"Warning: No predictions found for video {video_name}")
            continue
            
        gt_frames = ground_truths[video_name]
        pred_frames = predictions[video_name]
        
        # Process each frame based on ground truth
        for frame_name in gt_frames.keys():
            if frame_name not in pred_frames:
                print(f"Warning: No prediction found for {video_name}/{frame_name}")
                continue
            
            total_samples += 1
            
            try:
                # Load ground truth first
                gt_path = gt_frames[frame_name]
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    print(f"Warning: Cannot load ground truth {gt_path}")
                    continue
                
                # Load prediction
                pred_path = pred_frames[frame_name]
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                if pred is None:
                    print(f"Warning: Cannot load prediction {pred_path}")
                    continue
                
                # Resize prediction to match ground truth size (important!)
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Process ground truth: convert to float [0,1] then to uint8 [0,255] (same as original)
                #gt = gt.astype(np.float32) / 255.0  # Convert to [0,1] first
                #gt = np.asarray(gt * 255, np.uint8)  # Then scale to [0,255] and convert to uint8
                gt = np.asarray(gt, np.uint8)  # Simplified like test.py
                
                # Process prediction: normalize to [0,1] (same as original)
                pred = pred.astype(np.float32) / 255.0
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                
                # Calculate FM with float prediction (same as original)
                FM.step(pred=pred, gt=gt)
                
                # Convert prediction to uint8 for other metrics (same as original)
                pred_uint8 = (pred * 255).astype(np.uint8)
                SM.step(pred=pred_uint8, gt=gt)
                EM.step(pred=pred_uint8, gt=gt)
                MAE.step(pred=pred_uint8, gt=gt)
                
                matched_samples += 1
                
            except Exception as e:
                print(f"Error processing {video_name}/{frame_name}: {e}")
                continue
    
    if matched_samples == 0:
        raise ValueError("No valid prediction-ground truth pairs found!")
    
    print(f"Processed {matched_samples}/{total_samples} samples")
    
    # Get results
    fm_results = FM.get_results()
    fm = fm_results["fm"]["adaptive"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]["curve"].mean()
    mae = MAE.get_results()["mae"]
    
    return {
        "fm": fm,
        "sm": sm,
        "em": em,
        "mae": mae,
        "samples": matched_samples
    }

def main():
    args = parse_args()
    
    print(f"Loading predictions from: {args.pred_path}")
    predictions = load_citation_format_predictions(args.pred_path)
    print(f"Found {len(predictions)} videos in predictions")
    
    print(f"Loading ground truth from: {args.gt_path}")
    ground_truths = load_ground_truth(args.gt_path, args.dataset)
    print(f"Found {len(ground_truths)} videos in ground truth")
    
    print("Starting evaluation...")
    results = evaluate_predictions(predictions, ground_truths)
    
    # Print results
    time_now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"{time_now} Evaluation Results:")
    print(f"EM: {results['em']:.4f}, SM: {results['sm']:.4f}, F_beta: {results['fm']:.4f}, MAE: {results['mae']:.4f}")
    print(f"Total samples evaluated: {results['samples']}")
    

if __name__ == "__main__":
    main()