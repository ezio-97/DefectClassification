import os
import pandas as pd
from ultralytics import YOLO

# ==========================================
# 1. Configuration & Setup
# ==========================================

# AUTOMATICALLY find the dataset we just downloaded
# This points to: current_folder/gc10_yolo_dataset/data.yaml
DATASET_YAML = os.path.join(os.getcwd(), 'gc10_yolo_dataset', 'data.yaml')

# Check if it exists before starting
if not os.path.exists(DATASET_YAML):
    raise FileNotFoundError(f"Could not find dataset at {DATASET_YAML}. Did you run the previous script?")

RESULTS_DIR = os.path.join(os.getcwd(), 'detection_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Dataset Path: {DATASET_YAML}")
print(f"Results will be saved to: {RESULTS_DIR}")

# Models evaluated in the paper
YOLO_MODELS = [
    'yolov5m6u.pt',  # Ultra-real-time optimized
    'yolov8m.pt',    # Balanced accuracy/speed
    'yolov8x.pt',    # High accuracy
    'yolov9e.pt'     # Experimental, highest performance in paper
]

# Paper's optimal hyperparameters
HYPERPARAMS = {
    'epochs': 100,      # Set to 5 or 10 if you just want a quick test run!
    'batch': 16,        # Paper used 64, but 16 is safer for typical GPUs
    'imgsz': 640,
    'optimizer': 'Adam', 
    'lr0': 0.0001,
    'lrf': 0.2,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'augment': True      
}

# ==========================================
# 2. YOLO Training & Tuning Pipeline
# ==========================================
def train_yolo_models():
    model_metrics = []

    for model_name in YOLO_MODELS:
        print(f"\n>>> Starting Process for {model_name}...")
        
        # Initialize with pre-trained weights
        model = YOLO(model_name)

        # A. Hyperparameter Tuning 
        # WARNING: This takes a VERY long time (days). 
        # Run in a dedicated server.
        # print(f"Tuning {model_name}...")
        # model.tune(data=DATASET_YAML, epochs=30, iterations=50, optimizer='Adam')
        
        # B. Training with Optimal Parameters
        print(f"Training {model_name}...")
        
        # We try/except here so if one model fails (e.g. OOM), the others still run
        try:
            results = model.train(
                data=DATASET_YAML,
                project=RESULTS_DIR,
                name=model_name.replace('.pt', ''),
                **HYPERPARAMS
            )

            # C. Validation/Evaluation
            # Ultralytics automatically runs val after train, but we fetch metrics here
            metrics = model.val()
            
            # Extract Key Metrics used in paper: Precision, Recall, mAP@0.5
            model_metrics.append({
                'Model': model_name,
                'Precision': metrics.results_dict['metrics/precision(B)'],
                'Recall': metrics.results_dict['metrics/recall(B)'],
                'mAP50': metrics.results_dict['metrics/mAP50(B)'],
                'mAP50-95': metrics.results_dict['metrics/mAP50-95(B)'],
                'Weight_Path': str(results.save_dir / 'weights' / 'best.pt')
            })
            
        except Exception as e:
            print(f"!!! Error training {model_name}: {e}")

    return pd.DataFrame(model_metrics)

def select_best_model(metrics_df):
    """
    Selects the best model based on mAP@0.5, as emphasized in the paper results.
    """
    if metrics_df.empty:
        print("No models finished training successfully.")
        return None

    print("\n>>> Final Evaluation Results:")
    print(metrics_df.sort_values(by='mAP50', ascending=False))

    # Paper Conclusion: YOLOv9e was selected for highest mAP
    best_model = metrics_df.loc[metrics_df['mAP50'].idxmax()]
    
    print(f"\n>>> BEST MODEL SELECTED: {best_model['Model']}")
    print(f"    mAP@0.5: {best_model['mAP50']:.4f}")
    print(f"    Path: {best_model['Weight_Path']}")
    
    return best_model


if __name__ == '__main__':
    
    # 1. Run YOLO Pipeline
    results_df = train_yolo_models()
    
    # 2. Select Best
    if not results_df.empty:
        best_model_info = select_best_model(results_df)
        
        # 3. Save results to CSV for reporting
        csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Comparison saved to: {csv_path}")
    else:
        print("Training failed for all models.")

        