import numpy as np
import pandas as pd
import argparse
from pathlib import Path


npz_file = Path("D:/smsml_ratu-chairunisa/Membangun_model/processed/processed_data.npz")
out_dir = Path("D:/smsml_ratu-chairunisa/Membangun_model/processed/hasil_csv")

def npz_to_csv(npz_file, out_dir):
    data = np.load(npz_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Keys in NPZ:", data.files)

    for key in data.files:
        arr = data[key]
        if arr.ndim <= 2:
            df = pd.DataFrame(arr)
        else:
            df = pd.DataFrame(arr.reshape(arr.shape[0], -1))

        csv_path = out_dir / f"{key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {key} → {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path ke file .npz")
    parser.add_argument("--out", type=str, default="csv_output", help="Folder output CSV")
    args = parser.parse_args()

    npz_to_csv(args.file, args.out)
    print("Done.")