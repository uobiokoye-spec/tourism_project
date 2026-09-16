import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Dynamically drop either 'CustomerID' or 'Unnamed: 0' if present
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)
elif "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
