import os
import sys
import csv
import cv2
import numpy as np
import kitti_dataHandler
from pathlib import Path
import argparse
from sklearn.metrics import precision_score, recall_score


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainSetOutput',
        help='Set to True to output results on training set, set to False to only output on test',
        type = kitti_dataHandler.str2bool,
        default=False,
    )
    parser.add_argument(
        '--multipler',
        help='Set to True to output results on training set, set to False to only output on test',
        type = float,
        default=0.8,
    )
    args = parser.parse_args()

    if args.trainSetOutput:
        print("----Train Mode----")
        mode='train'
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
        multipliers = [0.6, 0.8, 1, 1.2]
    else:
        print("----Test Mode----")
        mode='test'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
        multipliers = [args.multipler]

    ################
    # Options
    ################
    # Input dir and output dir
    depth_dir = os.path.abspath(f'./data/{mode}/est_depth')
    gt_seg_dir = os.path.abspath('./data/train/gt_segmentation')

    ################
    for mult in multipliers:
        if args.trainSetOutput:
            output_dir = os.path.abspath(f'data/{mode}/est_segmentation/{mult}')
        else:
            output_dir = os.path.abspath(f'data/{mode}/est_segmentation')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cumulativePres = 0
        cumulativeRecall = 0
        for sample_name in sample_list:
            # Read depth map
            depthPath = depth_dir + '/' + sample_name + '.png'
            imgDepth = cv2.imread(depthPath, 0)
            # Discard depths less than 10cm from the camera

            # Create imageShaped matrix storing initially the value of 255
            mask = np.ones_like(imgDepth, dtype=np.uint8)*255
            # Read 2d bbox
            csvfile = open(f'bounding_box_{mode}.csv', newline='\n')
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                # Read 2d bbox
                if row[0] == sample_name:
                    x = int(row[1])
                    y = int(row[2])
                    w = int(row[3])
                    h = int(row[4])
                    # in some train cases, the bounding box top left corner is negative so fixed this
                    if x < 0:
                        w = w - abs(x)
                        x = 0
                    # in some train cases the bounding box extends beyond picture
                    if y+h > imgDepth.shape[0]:
                        h = imgDepth.shape[0]-y
                    temp = imgDepth[y:(y+h), x:(x+w)]
                    # Estimate the average depth of the objects
                    averageDepth = np.mean(temp)
                    # Measure standard deviation of 
                    std = np.std(temp)
                    # Find the pixels within a certain distance from the centroid
                    # set them equal to 0
                    for i in range(h):
                        for j in range(w):
                            if abs(imgDepth[y+i, x+j]-averageDepth) < mult*std:
                                mask[y+i, x+j] = 0

            if args.trainSetOutput:
                segPath = gt_seg_dir + '/' + sample_name + '.png'
                gt_seg_file = cv2.imread(segPath, 0)
                # Sets all car pixels to 0
                gt_seg_file[gt_seg_file<255] = 0
                precision = precision_score(gt_seg_file.flatten(), mask.flatten(), pos_label = 0)
                recall = recall_score(gt_seg_file.flatten(), mask.flatten(), pos_label = 0)
                cumulativePres += precision
                cumulativeRecall += recall
            # Save the segmentation mask      
            cv2.imwrite(output_dir + '/' + f'{sample_name}.png',mask)
        if args.trainSetOutput:
            print(f'standard dev mult: {mult}, precision: {cumulativePres/10:.3}, recall: {cumulativeRecall/10:.3}')

        
if __name__ == '__main__':
    main()
