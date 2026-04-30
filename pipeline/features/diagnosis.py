from datasets import load_dataset
import pandas as pd

ds = load_dataset("kdave/Indian_Financial_News", split="train")
df = pd.DataFrame(ds)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Dtypes:", df.dtypes)
print(df.head(3).to_string())
print("\nUnique label values:", df[df.columns[-1]].unique() if len(df.columns) > 1 else "N/A")