import json
from pathlib import Path
from data_generator import generate_dataset

ds = generate_dataset(60, 0.33, 42)
out_path = Path("outputs/synthetic_chargeback_dataset.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
print("Dataset re-generated successfully with CARD and UPI networks.")
