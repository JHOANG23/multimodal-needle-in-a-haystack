
import os
import pickle
import random
from PIL import Image
import base64
import json
from utils import load_image_paths


def main():
    # Load image paths from stitched images
    stitched_image_paths = load_image_paths(data_path)
    # load image paths from original images
    image_paths = load_image_paths('val2014')
    

    sequences = []
    # load meta data N_ROW_N_COL.json
    with open(os.path.join(meta_path, str(N_ROW) + '_'+str(N_COL)+ '.json'), 'r') as f:
        meta_data = json.load(f)

    # Generate image sequences
    for i in range(N_SEQUENCES):
        if SEQUENCE_LENGTH == 1:
            sequence = [stitched_image_paths[i]]
        else:
            sequence = random.sample(stitched_image_paths, SEQUENCE_LENGTH)
        target_paths = []
        idx_list = []
        row_list = []
        col_list = []
        if i < N_SEQUENCES/2:
            
            generated_j = set()
            for _ in range(N_NEEDLES):
                while True:
                    j = random.randint(0, SEQUENCE_LENGTH*N_COL*N_ROW-1)
                    if j not in generated_j or len(generated_j) >= SEQUENCE_LENGTH*N_COL*N_ROW:
                        generated_j.add(j)
                        break
                idx, loc = divmod(j, N_ROW*N_COL)
                row, col = divmod(loc, N_COL)
                stitched_path = sequence[idx]
                stitched_path = stitched_path.split('/')[-1]
                # locate the image path in the stitched image
                target_path = meta_data[stitched_path][str(row)+'_'+str(col)]
                #print(idx, row, col, stitched_path, target_path)
                target_paths.append(target_path)
                idx_list.append(idx)
                row_list.append(row)
                col_list.append(col)
        else:
            idx = -1
            row = col = -1
            # sample a path from the image_paths other than path in the sequence
            stitched_paths = [path.split('/')[-1] for path in sequence] 
            #exclude_images = meta_data[stitched_path].values()
            exclude_images = []
            for path in stitched_paths:
                exclude_images += meta_data[path].values()
            generated_path = set()
            for _ in range(N_NEEDLES):
                while True:
                    target_path = random.choice([path for path in image_paths if path not in exclude_images])
                    if target_path not in generated_path:
                        generated_path.add(target_path)
                        break
                target_paths.append(target_path)
                idx_list.append(idx)
                row_list.append(row)
                col_list.append(col)
        
        sequence_data = {
            'id': i,
            'image_ids': sequence,
            'index': idx_list,
            'row': row_list,
            'col': col_list,
            'target': target_paths
        }
        sequences.append(sequence_data)

    # Save sequences to JSON file
    with open(os.path.join(output_dir, output_json), 'w') as f:
        json.dump(sequences, f, indent=4)

if __name__ == "__main__":
    random.seed(0)
    SEQUENCE_LENGTH = 2  # Length of each image sequence
    N_SEQUENCES = 1000 # Number of sequences to generate
    N_ROW = N_COL = 2
    N_NEEDLES = 2
    data_dir = 'images_stitched'
    meta_path = 'metadata_stitched'

    res_dir = str(N_ROW)+'_'+ str(N_COL)
    data_path = os.path.join(data_dir, res_dir)
    output_dir = 'metadata_stitched'
    output_json = str(N_NEEDLES) + '_' + 'annotations_'+ str(SEQUENCE_LENGTH) + '_' + res_dir +'.json'
    main()
