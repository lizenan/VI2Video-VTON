from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import cv2
import numpy as np
import mediapipe as mp
from segment_anything import SamPredictor, sam_model_registry

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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



model_path = args.model_path

sam = sam_model_registry['vit_h'](checkpoint="/mnt/data/vivid_ckpts/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)


if __name__ == '__main__':
    #count = 0
    
    input_video = cv2.VideoCapture(model_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_width = 624#int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = 832#int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = f'/home/zenan/ViViD/heygen_hand_detection.mp4'
    output_video = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height)) #(768,1024)
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        final_mask = np.zeros((frame_height,frame_width)) #* 255
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            # Process the frame and detect hands
            results = hands.process(image_rgb)

            # If hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the bounding box by finding the min and max x and y coordinates of hand landmarks
                    h, w, _ = frame.shape
                    x_min = w
                    y_min = h
                    x_max = y_max = 0

                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        if x < x_min:
                            x_min = x
                        if y < y_min:
                            y_min = y
                        if x > x_max:
                            x_max = x
                        if y > y_max:
                            y_max = y

                    # Draw the bounding box around the hand
                    #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    box_width = (x_max-x_min)*.2
                    box_height = (y_max-y_min)*.2
                    input_box = np.array([x_min-box_width, y_min-box_height, x_max+box_width, y_max+box_height])
                    predictor.set_image(frame)
                    masks, _, _ = predictor.predict(box=input_box,multimask_output=False)
                    binary_mask = masks[0].astype(np.uint8) #* 255
                    final_mask[:] += binary_mask
                    #mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                    #colored_mask = np.zeros_like(frame)
                    #colored_mask[:, :] = [0, 0, 255]  # Red color (BGR format)

                    # Apply the mask onto the colored image
                    #masked_image = cv2.bitwise_and(colored_mask, mask_3channel)
                    #alpha = 0.6  # Transparency factor
                    #frame = cv2.addWeighted(frame, 1, masked_image, alpha, 0)

        # Display the frame
        final_mask = (1-final_mask) * 255
        mask_3channel = cv2.cvtColor(final_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        output_video.write(mask_3channel)
        
    input_video.release()
    output_video.release()
    #cv2.destroyAllWindows()


    

