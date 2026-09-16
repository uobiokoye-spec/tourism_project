import pandas as pd
from huggingface_hub import HfApi
import os

RAW_PATH = "tourism_project/data/tourism.csv"
DATASET_REPO_NAME = "tourism_dataset"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset validated successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())

# Upload dataset to Hugging Face Hub
try:
    hf_api = HfApi(token=os.environ.get("HF_TOKEN"))
    hf_api.create_repo(repo_id=DATASET_REPO_NAME, repo_type="dataset", exist_ok=True)
    
    temp_csv_path = "temp_tourism.csv"
    df.to_csv(temp_csv_path, index=False)
    hf_api.upload_file(
        path_or_fileobj=temp_csv_path,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_NAME,
        repo_type="dataset",
    )
    os.remove(temp_csv_path)
    print(f"Dataset uploaded to Hugging Face Hub: {DATASET_REPO_NAME}/tourism.csv")
except Exception as e:
    print(f"Failed to upload dataset to Hugging Face Hub: {e}")
    print("Please ensure HF_TOKEN is set as a secret in your GitHub repository and has write access.")
