### Requirements

- conda
- Python 3.7 or higher
- open-cv version 4.4.0.46

### Installation

```bash
# If you need to install conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create env with requirements
conda env create -f environment.yml

# activate the conda environment
conda activate tempAss3

```

### Download yolo weights
```bash
cd yolo
wget https://pjreddie.com/media/files/yolov3.weights
```

### Running code
```bash
# Estimates Depth on Test Set
python part1_estimate_depth.py

# Estimates Depth on Training Set
python part1_estimate_depth.py --trainSetOutput True

# Performs Object Detection on Test Set and saves bounding box
python part2_yolo.py

# Performs Object Detection on Train Set and saves bounding box
python part2_yolo.py --trainSetOutput True

# Performs Instance Segmentation on Test Set (must run part2_yolo.py prior)
python part3_segmentation.py 

# Performs Instance Segmentation on Train Set and evaluates 4 different distance thresholds (must run part2_yolo.py prior)
python part3_segmentation.py --trainSetOutput True

# Performs Instance Segmentation on Test Set with a multiplier of 2 (must run part2_yolo.py prior)
python part3_segmentation.py --multipler 2
```
