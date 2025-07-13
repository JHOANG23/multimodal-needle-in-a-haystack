import os
import pickle
import random
from PIL import Image
import base64
import json
from utils import load_image_paths

def get_all_images(root_dir):
    stitched_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                stitched_paths.append(os.path.join(subdir, file))
    return stitched_paths


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))   # e.g. .../CS228/multimodal-needle-in-a-haystack
    cs228_root = os.path.abspath(os.path.join(current_dir, '..'))

    stitched_image_paths_dir = os.path.join(cs228_root, 'multimodal-needle-in-a-haystack', 'images_stitched')
    stitched_image_paths = get_all_images(stitched_image_paths_dir)

    val2014_images_dir = os.path.join(cs228_root, 'coco', 'images', 'val2014', 'val2014')
    image_paths = get_all_images(val2014_images_dir)

    sequences = []
    # load meta data N_ROW_N_COL.json
    with open(os.path.join(meta_path, str(N_ROW) + '_'+str(N_COL)+ '.json'), 'r') as f:
        meta_data = json.load(f)
        print(f"opened {N_ROW}_{N_COL}.json successfully\n")

    # Generate image sequences
    for i in range(N_SEQUENCES):
        if SEQUENCE_LENGTH == 1:
            sequence = [stitched_image_paths[i]]
            print(f"sequence: {sequence}")
        else:
            sequence = random.sample(stitched_image_paths, SEQUENCE_LENGTH)

        if i < N_SEQUENCES/2:
            j = random.randint(0, SEQUENCE_LENGTH*N_COL*N_ROW-1)
            idx, loc = divmod(j, N_ROW*N_COL)
            row, col = divmod(loc, N_COL)
            print(f"idx: {idx}")
            stitched_path = sequence[idx]
            stitched_path = os.path.basename(sequence[idx])
            print(f"stitched_path: {stitched_path}")
            target_path = meta_data[stitched_path][str(row)+'_'+str(col)]
        else:
            idx = -1
            row = col = -1
            stitched_paths = [path.split('/')[-1] for path in sequence] 
            exclude_images = []
            for path in stitched_paths:
                exclude_images += meta_data[os.path.basename(path)].values()
            target_path = random.choice([path for path in image_paths if path not in exclude_images])

        # --- Begin: convert absolute paths to relative paths ---

        # Convert stitched image paths to relative (keep after images_stitched/)
        sequence_relative = []
        for p in sequence:
            parts = p.replace("\\", "/").split("images_stitched/", 1)
            if len(parts) > 1:
                rel_path = "images_stitched/" + parts[1]
            else:
                rel_path = p.replace("\\", "/")  # fallback
            sequence_relative.append(rel_path)

        # Convert target path to relative (keep after val2014/)
        if i < N_SEQUENCES/2:
            # target_path from meta_data is just filename like 'COCO_val2014_000000243218.jpg'
            target_path = "val2014/" + target_path.replace("\\", "/")
        else:
            parts = target_path.replace("\\", "/").split("val2014/", 1)
            if len(parts) > 1:
                target_path = "val2014/" + parts[1]
            else:
                target_path = target_path.replace("\\", "/")

        # --- End convert paths ---

        sequence_data = {
            'id': i,
            'image_ids': sequence_relative,
            'index': idx,
            'row': row,
            'col': col,
            'target': target_path
        }
        sequences.append(sequence_data)

    # Save sequences to JSON file
    with open(os.path.join(output_dir, output_json), 'w') as f:
        json.dump(sequences, f, indent=4)

if __name__ == "__main__":
    random.seed(0)
    SEQUENCE_LENGTH = 1  # Length of each image sequence (DONT CHANGE)
    N_SEQUENCES = 10000  # Number of sequences to generate
    N_ROW = int(os.getenv('N_ROW', '2'))
    N_COL = int(os.getenv('N_COL', '2'))
    data_dir = 'images_stitched'
    meta_path = 'metadata_stitched'

    res_dir = str(N_ROW)+'_'+ str(N_COL)
    data_path = os.path.join(data_dir, res_dir)
    output_dir = 'metadata_stitched'
    # output_json = 'annotations_'+ str(N_SEQUENCES) + '_' + res_dir +'.json'
    output_json = 'annotations_'+ str(SEQUENCE_LENGTH) + '_' + res_dir +'.json'
    main()
