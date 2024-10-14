from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed


if __name__ == '__main__':
    #count = 0
    input_video = cv2.VideoCapture(model_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_width = 624#int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = 832#int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = f'/home/zenan/ViViD/heygen2.mp4'
    output_video = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height)) #(768,1024)
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        edited_frame = cv2.resize(frame, (frame_width, frame_height))
        #masked_vton_img.save('/home/zenan/ViViD/OOTDiffusion/run/images_output/mask_test.jpg')
        #break
        output_video.write(edited_frame)
        
    input_video.release()
    output_video.release()
    #cv2.destroyAllWindows()


    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    

