import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
import kitti_dataHandler
import argparse
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainSetOutput',
        help='Set to True to output results on training set, set to False to only output on test',
        type = kitti_dataHandler.str2bool,
        default=False,
    )
    args = parser.parse_args()
    if args.trainSetOutput:
        print("----Train Mode----")
        mode='train'
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        print("----Test Mode----")
        mode='test'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################
    # Options
    ################
    # Input dir and output dir
    disp_dir = os.path.abspath(f'./data/{mode}/disparity')
    output_dir = os.path.abspath(f'./data/{mode}/est_depth')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    calib_dir = os.path.abspath(f'./data/{mode}/calib')

    
    ################

    for sample_name in (sample_list):
        # Read disparity map
        disparityPath = disp_dir + '/' + sample_name + '.png'
        imgDisparity = cv.imread(disparityPath, 0)
        # Read calibration info
        calib_text_path = calib_dir +'/' + sample_name + '.txt'
        frame_calib = kitti_dataHandler.read_frame_calib(calib_text_path)
        calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
        baseline = calib.baseline
        focal_length = calib.f
        # Calculate depth (z = f*B/disp) & discard pixels past 80m or less than 10cm
        depth = kitti_dataHandler.calculate_depth(imgDisparity, baseline, focal_length)
        
        # Save depth map
        cv.imwrite(output_dir + '/' + f'{sample_name}.png', depth)


if __name__ == '__main__':
    main()
