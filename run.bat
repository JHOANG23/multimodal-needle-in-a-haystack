@echo off

REM Set how many stitched images you want to make (DON'T CHANGE)
set N_IMG = 10000 

REM Set the number of rows and columns for stitched images. Don't change! Model can barely handle 2x2 stitched images as is
set N_ROW=2
set N_COL=2

REM Set the number of targets per quadrant that you want. Helps keep the evaluations balanced.
REM SEQ_LENGTH=50 means we run evals 200 times (50*4). 4 for each quadrant
set SEQ_LENGTH=50

echo Running sample_stitched_images.py with N_ROW=%N_ROW% and N_COL=%N_COL%

python sample_stitched_images.py
python sample_single_needle.py
python generate_balanced_annotations.py

REM 
pause
