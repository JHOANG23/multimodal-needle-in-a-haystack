import json
import random
import os
from collections import defaultdict

# === Config ===
SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', '1'))  # Number of targets per quadrant
N_ROW = int(os.getenv('N_ROW', '2'))
N_COL = int(os.getenv('N_COL', '2'))
INPUT_FILE = 'metadata_stitched/annotations_1_2_2.json'
OUTPUT_FILE = f'metadata_stitched/annotations_balanced_{SEQ_LENGTH}_2_2.json'

random.seed(42)  # For reproducibility

# === Load original metadata ===
with open(INPUT_FILE, 'r') as f:
    original_data = json.load(f)

# === Organize entries by quadrant ===
quadrant_buckets = defaultdict(list)  # Key = (row, col), Value = list of entries

for entry in original_data:
    row = entry['row']
    col = entry['col']
    quadrant_buckets[(row, col)].append(entry)

# === Sample SEQ_LENGTH entries per quadrant ===
balanced_data = []
for row in range(N_ROW):
    for col in range(N_COL):
        key = (row, col)
        if key not in quadrant_buckets:
            raise ValueError(f"No data for quadrant (row={row}, col={col})")
        
        entries = quadrant_buckets[key]
        if len(entries) < SEQ_LENGTH:
            raise ValueError(f"Not enough entries in quadrant {key} to sample {SEQ_LENGTH}")
        
        sampled = random.sample(entries, SEQ_LENGTH)
        balanced_data.extend(sampled)

# === Assign new integer keys for the balanced dataset ===
balanced_dict = {i: entry for i, entry in enumerate(balanced_data)}

# === Save to new JSON file ===
with open(OUTPUT_FILE, 'w') as f:
    json.dump(balanced_dict, f, indent=4)

print(f"Saved balanced metadata to: {OUTPUT_FILE}")
