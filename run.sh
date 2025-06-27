#!/bin/bash

# Set how many stitched images you want to make (DON'T CHANGE)
export N_IMG=10000

# Set the number of rows and columns for stitched images. Don't change!
# Model can barely handle 2x2 stitched images as is
export N_ROW=2
export N_COL=2

# Set the number of targets per quadrant that you want. Helps keep the evaluations balanced.
# SEQ_LENGTH=50 means we run evals 200 times (50*4). 4 for each quadrant
export SEQ_LENGTH=50

echo "Running sample_stitched_images.py with N_ROW=$N_ROW and N_COL=$N_COL and SEQ_LENGTH=$SEQ_LENGTH"

python sample_stitched_images.py
python sample_single_needle.py
python generate_balanced_annotations.py

read -p "Press enter to exit..."
