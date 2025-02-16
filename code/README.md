# Regarding datasets
- only the files necessary for the pipeline are in this submission
- all 6 datasets would be too big for the submission (2GB)
# Setting Up the Virtual Environment
- necessary imports:
```python
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2 as cv
import math
import pytesseract
import json
```
- the following versions were used (especially tensorflow has to have the right version):
```bash
pandas-version: 2.15.0
numpy-version: 1.26.4
tensorflow-version: 2.15.0
opencv-version: 4.8.1
pytesseract-version: 0.3.13
```
- thus, for setting up the environment with conda, we need the following command:
```bash
conda create -n ipda -c conda-forge pandas numpy tensorflow=2.15 opencv=4.8 matplotlib pytesseract
```
- afterwards, we still need to install the jupyter kernel to execute the files in ipynb_files_for_each_task
```bash
conda activate ipda
conda install jupyter
```
# Explaining the Code
- executing the pipeline can easily be done by executing the `main.py`-file
    - if debug is set to False (*see line 23*), the output will show recall and precision on found words, based on dataset for task 6
    - if debug is set to True, the output in the console will still be the same but there will also be images saved to the `predicted_images`-folder, displaying the following images:
        1. predicted segmentation (task 2)
        2. opening performed on predicted segmentation (pre-processing step task 3)
        3. canny edge detection performed on the opened segmentation (pre-processing step task 3)
        4. closing performed on predicted segmentation (testing)
        5. rotated and cropped segmentation (task 3)
        6. cleaned and binarized version of the rotated and cropped segmentation (task 4)
- executing the bonus task for task 5 can easily be done by executing the `main_task_5_bonus.py`-file
    - if debug is set to False (*see line 12*), the output will show recall and precision on found words for cleaned and uncleaned images, based on the dataset for task 3
    - if debug is set to True, the output in the console will still be the same but there will also be images saved to the `predicted_images`-folder, displaying the following images:
        1. rotated image (task 3)
        2. cleaned and binarized version of the rotated image (task 4)