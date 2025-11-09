import pandas as pd
import json
import os

# Adjusted input and output paths
input_file = "data/Gen_AI Dataset.xlsx"
output_file = "data/shl_assessments.json"

# Load Excel file
print(f"📘 Loading dataset from: {input_file}")
df = pd.read_excel(input_file)

# Normalize column names (make them lowercase and underscore-separated)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print(f"✅ Columns found: {list(df.columns)}")

# Try to detect likely columns automatically
possible_name_cols = [c for c in df.columns if "name" in c or "assessment" in c]
possible_url_cols = [c for c in df.columns if "url" in c or "link" in c]
possible_desc_cols = [c for c in df.columns if "desc" in c]
possible_type_cols = [c for c in df.columns if "type" in c]

def safe_get(row, options):
    for c in options:
        if c in row and pd.notnull(row[c]):
            return str(row[c])
    return ""

records = []
for _, row in df.iterrows():
    record = {
        "name": safe_get(row, possible_name_cols),
        "url": safe_get(row, possible_url_cols),
        "description": safe_get(row, possible_desc_cols),
        "test_type": safe_get(row, possible_type_cols),
    }
    if record["name"]:
        records.append(record)

os.makedirs("data", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\n✅ Created {output_file} with {len(records)} assessments.")
