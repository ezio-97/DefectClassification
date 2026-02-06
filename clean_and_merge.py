import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================

# 1. The Output Directory (Where the merged dataset will go)
OUTPUT_DIR = Path(r"D:\Backup\WORK\MACH-3D\DefectClassification\Cleaned_dataset")

# 2. The Master Class List (Target Schema) - ORDER MATTERS
TARGET_CLASSES = [
    'blobs',           # 0
    'cracks',          # 1
    'over_extrusion',  # 2
    'spaghetti',       # 3
    'stringing',       # 4
    'under_extrusion', # 5
    'layer_shift',     # 6
    'warp'             # 7
]

# 3. Typo and Alias Correction Map
# Maps input class names to the TARGET_CLASSES names.
# Any class NOT in TARGET_CLASSES and NOT in this map will be dropped.
NAME_MAPPING = {
    'warping': 'warp',
    'warpping': 'warp',
    'wrap': 'warp',
    'Shift': 'layer_shift',
    'shift': 'layer_shift'
}

# 4. Input Datasets Configuration
# Add the root path for each dataset and its specific 'names' list from its data.yaml.
DATASETS = [
    {
        "name": "master_dataset",
        "path": Path(r"D:\Backup\WORK\MACH-3D\DefectClassification\FINAL_MERGED_DATASET"), 
        "classes": ['blobs', 'cracks', 'over_extrusion', 'spaghetti', 'stringing', 'under_extrusion', 'layer_shift', 'warp']
    },
    {
        "name": "stringing_ds",
        "path": Path(r"D:\Backup\WORK\MACH-3D\DefectClassification\stringing defect 3d printing.v1i.yolov8"), # UPDATE THIS PATH
        "classes": ['stringing']
    },
    {
        "name": "warping_ds_1",
        "path": Path(r"D:\Backup\WORK\MACH-3D\DefectClassification\warping"), # UPDATE THIS PATH
        "classes": ['warping']
    },
    {
        "name": "warping_ds_2",
        "path": Path(r"D:\Backup\WORK\MACH-3D\DefectClassification\warping 3d prints.v1"), # UPDATE THIS PATH
        "classes": ['warpping']
    },
    {
        "name": "layer_shift_ds",
        "path": Path(r"D:\Backup\WORK\MACH-3D\DefectClassification\layer shifting.v1i.yolov8"), # UPDATE THIS PATH
        "classes": ['Nozzle', 'Shift', 'part', 'shift', 'wrap']
    }
]

# ================= HELPER FUNCTIONS =================

def get_target_id(source_class_name):
    """Returns the target class ID or None if class should be ignored."""
    # 1. Check exact match
    if source_class_name in TARGET_CLASSES:
        return TARGET_CLASSES.index(source_class_name)
    
    # 2. Check alias/typo map
    if source_class_name in NAME_MAPPING:
        target_name = NAME_MAPPING[source_class_name]
        if target_name in TARGET_CLASSES:
            return TARGET_CLASSES.index(target_name)
            
    # 3. If no match (e.g., 'Nozzle', 'part'), return None
    return None

def polygon_to_bbox(coords):
    """Converts normalized polygon coordinates [x1, y1, x2, y2...] to normalized bbox [cx, cy, w, h]."""
    xs = coords[0::2]
    ys = coords[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    w = max_x - min_x
    h = max_y - min_y
    cx = min_x + (w / 2)
    cy = min_y + (h / 2)
    
    return [cx, cy, w, h]

def process_dataset(dataset_cfg, image_pool, label_pool):
    """Reads dataset, fixes labels, and copies to pool."""
    print(f"Processing {dataset_cfg['name']}...")
    
    # Construct mapping for this specific dataset's IDs to Target IDs
    id_map = {}
    for local_id, class_name in enumerate(dataset_cfg['classes']):
        target_id = get_target_id(class_name)
        id_map[local_id] = target_id 
        if target_id is None:
            print(f"  [Info] Dropping class '{class_name}' (ID {local_id})")

    # Walk through standard YOLO folder structure (train/val/test)
    root = dataset_cfg['path']
    # Look recursively for images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [p for p in root.rglob('*') if p.suffix.lower() in image_extensions]
    
    for img_path in tqdm(images):
        # Derive label path (YOLO convention: images/ -> labels/)
        # We try to find the label file by replacing 'images' with 'labels' in path
        # and changing suffix to .txt
        
        parts = list(img_path.parts)
        try:
            # Simple heuristic: find 'images' folder and swap to 'labels'
            idx = len(parts) - 1 - parts[::-1].index('images')
            parts[idx] = 'labels'
            label_path = Path(*parts).with_suffix('.txt')
        except ValueError:
            # If folder structure isn't standard images/labels, assume adjacent
            label_path = img_path.with_suffix('.txt')

        # Generate unique filename to avoid collisions
        unique_name = f"{dataset_cfg['name']}_{img_path.name}"
        
        new_label_content = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                data = line.strip().split()
                if not data: continue
                
                cls_id = int(data[0])
                coords = [float(x) for x in data[1:]]
                
                # 1. Map Class ID
                new_id = id_map.get(cls_id)
                if new_id is None:
                    continue # Skip unwanted classes
                
                # 2. Fix Box vs Segmentation (Delete tags / Convert to Box)
                # If coords > 4, it's a polygon. We convert to bbox.
                if len(coords) > 4:
                    coords = polygon_to_bbox(coords)
                
                # Formulate new line
                new_line = f"{new_id} {' '.join(map(str, coords))}\n"
                new_label_content.append(new_line)
        
        # Only save if we have an image (labels are optional, but if we filter all labels out, 
        # we might want to keep the image as background or drop it. 
        # Here we keep image even if empty labels, which is good for training FP reduction)
        
        # Copy Image
        shutil.copy2(img_path, image_pool / unique_name)
        
        # Write Label
        if new_label_content:
            with open(label_pool / unique_name.replace(img_path.suffix, '.txt'), 'w') as f:
                f.writelines(new_label_content)
        else:
            # Create empty label file if no valid labels exist
            (label_pool / unique_name.replace(img_path.suffix, '.txt')).touch()

def split_and_organize(pool_img, pool_lbl, final_base):
    """Splits pooled files into train/val/test."""
    print("Splitting and organizing data...")
    
    images = list(pool_img.glob('*'))
    random.shuffle(images)
    
    total = len(images)
    n_train = int(total * 0.7)
    n_val = int(total * 0.2)
    # n_test is remainder
    
    sets = {
        'train': images[:n_train],
        'valid': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    for split, split_imgs in sets.items():
        # Make directories
        (final_base / split / 'images').mkdir(parents=True, exist_ok=True)
        (final_base / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(split_imgs, desc=f"Moving {split}"):
            # Move Image
            dst_img = final_base / split / 'images' / img_path.name
            shutil.move(str(img_path), str(dst_img))
            
            # Move Label
            lbl_name = img_path.with_suffix('.txt').name
            src_lbl = pool_lbl / lbl_name
            dst_lbl = final_base / split / 'labels' / lbl_name
            
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))

# ================= EXECUTION =================

if __name__ == "__main__":
    # Create temp pool directories
    POOL_DIR = OUTPUT_DIR / "temp_pool"
    POOL_IMG = POOL_DIR / "images"
    POOL_LBL = POOL_DIR / "labels"
    
    # Clean start
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    POOL_IMG.mkdir(parents=True)
    POOL_LBL.mkdir(parents=True)
    
    # Process all datasets
    for ds in DATASETS:
        process_dataset(ds, POOL_IMG, POOL_LBL)
        
    # Split into final structure
    split_and_organize(POOL_IMG, POOL_LBL, OUTPUT_DIR)
    
    # Remove temp pool
    shutil.rmtree(POOL_DIR)
    
    # Create final data.yaml
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(TARGET_CLASSES)},
        'nc': len(TARGET_CLASSES)
    }
    
    with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"\nSUCCESS! Merged dataset created at: {OUTPUT_DIR}")
    print("Check the 'data.yaml' file in that folder to ensure paths are correct.")