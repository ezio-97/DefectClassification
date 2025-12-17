import kagglehub

# Download latest version
path = kagglehub.dataset_download("wengmhu/fdm-3d-printing-defect-dataset")

print("Path to dataset files:", path)