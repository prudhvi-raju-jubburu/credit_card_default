import kagglehub

# Download latest version
path = kagglehub.dataset_download("ifeanyichukwunwobodo/credit-card-default")

print("Path to dataset files:", path)