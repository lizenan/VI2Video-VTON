from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
parser = argparse.ArgumentParser(description='resize video')
parser.add_argument('--i', type=str, default="", required=True)
parser.add_argument('--o', type=str, default="", required=True)
args = parser.parse_args()

input_path = args.i
output_path = args.o

if __name__ == '__main__':
    #count = 0
    
    input_video = cv2.VideoCapture(input_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    ori_frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_frame_width = 624#
    desired_frame_height = 832#
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (desired_frame_width, desired_frame_height)) #(768,1024)
    original_aspect_ratio = ori_frame_width / ori_frame_height
    target_aspect_ratio = desired_frame_width / desired_frame_height
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        if original_aspect_ratio > target_aspect_ratio:
            new_width = int(ori_frame_height * target_aspect_ratio)
            x_offset = (ori_frame_width - new_width) // 2
            cropped_frame = frame[:, x_offset:x_offset + new_width]
        else:
            new_height = int(ori_frame_width / target_aspect_ratio)
            y_offset = (ori_frame_height - new_height) // 2
            cropped_frame = frame[y_offset:y_offset + new_height, :]
        
        resized_frame = cv2.resize(cropped_frame, (desired_frame_width, desired_frame_height))
        output_video.write(resized_frame)
        
    input_video.release()
    output_video.release()
    #cv2.destroyAllWindows()


    

