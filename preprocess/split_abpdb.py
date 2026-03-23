import argparse
import os
import numpy as np


REQUIRED_FILES = [
    "seq.npy",
    "phipsi.npy",
    "DSSP.npy",
    "distance_value.npy",
    "movement_vector.npy",
    "quater_number.npy",
    "mask.npy",
    "label.npy",
]


def ensure_required_files(input_dir):
    missing = []
    for filename in REQUIRED_FILES:
        full_path = os.path.join(input_dir, filename)
        if not os.path.exists(full_path):
            missing.append(full_path)
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")


def main():
    parser = argparse.ArgumentParser(description="Split ABPDB npy dataset into ABPDB_7 and ABPDB_3")
    parser.add_argument("--input_dir", type=str, default="./data/ABPDB/", help="Directory containing original ABPDB *.npy files")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for shuffling")
    args = parser.parse_args()

    input_dir = args.input_dir
    train_dir = os.path.join(input_dir, "ABPDB_7")
    test_dir = os.path.join(input_dir, "ABPDB_3")

    ensure_required_files(input_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    arrays = {}
    sample_count = None
    for filename in REQUIRED_FILES:
        full_path = os.path.join(input_dir, filename)
        arr = np.load(full_path)
        arrays[filename] = arr
        if sample_count is None:
            sample_count = arr.shape[0]
        elif arr.shape[0] != sample_count:
            raise ValueError(f"First dimension mismatch in {filename}: expected {sample_count}, got {arr.shape[0]}")

    indices = np.arange(sample_count)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    split_idx = int(sample_count * args.train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    for filename in REQUIRED_FILES:
        arr = arrays[filename]
        np.save(os.path.join(train_dir, filename), arr[train_idx])
        np.save(os.path.join(test_dir, filename), arr[test_idx])

    print(f"Total samples: {sample_count}")
    print(f"Train samples ({args.train_ratio:.2f}): {len(train_idx)} -> {train_dir}")
    print(f"Test samples ({1.0 - args.train_ratio:.2f}): {len(test_idx)} -> {test_dir}")
    print("Split completed.")


if __name__ == "__main__":
    main()
