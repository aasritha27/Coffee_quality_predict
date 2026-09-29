"""Merge arabica + robusta CQI files into data/raw/merged_data.csv (harmonised columns)."""
import pandas as pd
from common import ROOT

raw = ROOT / "data" / "raw"
ara = pd.read_csv(raw / "arabica_data_cleaned.csv", index_col=0)
rob = pd.read_csv(raw / "robusta_data_cleaned.csv", index_col=0)
rob = rob.rename(columns={
    "Fragrance...Aroma": "Aroma", "Salt...Acid": "Acidity", "Mouthfeel": "Body",
    "Uniform.Cup": "Uniformity", "Bitter...Sweet": "Sweetness"})
merged = pd.concat([ara, rob], ignore_index=True)
merged.to_csv(raw / "merged_data.csv", index=False)
print("merged_data.csv:", merged.shape, merged["Species"].value_counts().to_dict())
