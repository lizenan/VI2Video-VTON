import os
from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import cv2
import numpy as np
import mediapipe as mp
from segment_anything import SamPredictor, sam_model_registry
import onnxruntime as ort
import os.path as osp
import json
try:
    import torch
    from torchvision.ops import nms
except Exception as e:
    print(e)

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--cloth_path', type=str, default="", required=True)
parser.add_argument('--output_cloth_path', type=str, default="", required=True)
parser.add_argument('--onnx', default="", help='onnx file', required=True)
parser.add_argument('--text', default="",help='detecting texts (str or json), should be consistent with the ONNX model', required=True)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
parser.add_argument('--onnx-nms',default=True,action='store_false',help='whether ONNX model contains NMS and postprocessing', required=False)
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

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
sam = sam_model_registry['vit_h'](checkpoint="/mnt/data/vivid_ckpts/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

if model_type == 'hd' and category != 0:
    raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

def preprocess(image, size=(640, 640)):
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    image = cv2.resize(pad_image, size,
                       interpolation=cv2.INTER_LINEAR).astype('float32')
    image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)

def inference(ort_session,
              image_path,
              texts,
              size=(640, 640),
              **kwargs):
    # normal export
    # with NMS and postprocessing
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["num_dets", "labels", "scores", "boxes"],
                              {"images": input_ort})
    num_dets, labels, scores, bboxes = results
    num_dets = num_dets[0][0]
    labels = labels[0, :num_dets]
    scores = scores[0, :num_dets]
    bboxes = bboxes[0, :num_dets]
    if num_dets>0:
        bboxes = bboxes[0:1]
        bboxes /= scale_factor
        bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
        bboxes[:,0::2] = np.clip(bboxes[:, 0::2], 0, w)
        bboxes[:,1::2] = np.clip(bboxes[:, 1::2], 0, h)
        bboxes = bboxes.round().astype('int')
        return bboxes[0]
    
if __name__ == '__main__':
    #count = 0
    filename_without_ext = os.path.splitext(os.path.basename(model_path))[0]
    input_video = cv2.VideoCapture(model_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_mask_name = f"./{filename_without_ext}_agnostic_mask.mp4" #_{category_dict[category]}
    video_name = f'./{filename_without_ext}_agnostic.mp4' #_{category_dict[category]}
    output_mask = cv2.VideoWriter(video_mask_name, fourcc, fps, (frame_width, frame_height)) #(768,1024)
    output_video = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height)) #(768,1024)
    #output_hands_video = cv2.VideoWriter("hand_tests.mp4", fourcc, fps, (frame_width, frame_height))
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(768, 1024))
        final_hands_mask = np.zeros((1024,768))
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            # Process the frame and detect hands
            results = hands.process(frame)

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
                    input_box = np.array([max(x_min-box_width,0), max(y_min-box_height, 0), min(x_max+box_width,w), min(y_max+box_height,h)])
                    predictor.set_image(frame)
                    masks, _, _ = predictor.predict(box=input_box,multimask_output=False)
                    binary_mask = masks[0].astype(np.uint8) #* 255
                    final_hands_mask[:] += binary_mask

        final_hands_mask = ((1-final_hands_mask) * 255).astype(np.uint8)
        #mask_3channel = cv2.resize(cv2.cvtColor(final_hands_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR),(frame_width, frame_height))
        #output_hands_video.write(mask_3channel)
        model_img = Image.fromarray(frame)
        #cloth_img = Image.open(cloth_path).resize((768, 1024))
        #model_img = Image.fromarray(img).resize((768, 1024))
        keypoints = openpose_model(model_img.resize((384, 512)))
        model_parse, _ = parsing_model(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask = cv2.bitwise_and(np.array(mask), final_hands_mask)
        mask = Image.fromarray(mask)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        mask_gray = cv2.bitwise_and(np.array(mask_gray), (final_hands_mask/2).astype(np.uint8))
        mask_gray = Image.fromarray(mask_gray)
        #mask_gray.save("/home/zenan/ViViD/test_mask_gray.jpg")
        #mask.save("/home/zenan/ViViD/test_mask.jpg")
        masked_vton_img = Image.composite(mask_gray, model_img, mask)#.resize((frame_width, frame_height))
        edited_frame = cv2.cvtColor(np.array(masked_vton_img), cv2.COLOR_RGB2BGR)
        edited_frame = cv2.resize(edited_frame, (frame_width, frame_height))
        edited_mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
        edited_mask = cv2.resize(edited_mask, (frame_width, frame_height))
        #masked_vton_img.save('/home/zenan/ViViD/OOTDiffusion/run/images_output/mask_test.jpg')
        #break
        output_video.write(edited_frame)
        output_mask.write(edited_mask)
        
    input_video.release()
    output_video.release()
    #output_hands_video.release()
    output_mask.release()
    #cv2.destroyAllWindows()

    onnx_file = args.onnx
    # init ONNX session
    ort_session = ort.InferenceSession(
    onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Init ONNX Runtime session")
    if not osp.isfile(cloth_path):
        images = [
            osp.join(cloth_path, img) for img in os.listdir(cloth_path)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [cloth_path]

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines]
    elif args.text.endswith('.json'):
        texts = json.load(open(args.text))
    else:
        texts = [[t.strip()] for t in args.text.split(',')]

    print("Start to inference.")

    if args.onnx_nms:
        inference_func = inference

    for cloth in images:
        input_box = inference_func(ort_session, cloth, texts)
        
        cloth_path = cloth
        cloth_img = np.array(Image.open(cloth_path))
        h, w, _ = cloth_img.shape

        predictor.set_image(cloth_img)
        masks, _, _ = predictor.predict(box=input_box,multimask_output=False)
        binary_mask = masks[0].astype(np.uint8) * 255
        binary_mask = cv2.cvtColor(binary_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #cv2.imwrite("orig_mask.jpg",binary_mask)
        desired_width = 768
        desired_height = 1024
        #desired_mask = np.ones((1024,768,3)).astype(np.uint8)*255
        #desired_cloth = np.ones((1024,768,3)).astype(np.uint8)*255
        box_height = int((input_box[3]-input_box[1])*.1)
        box_width = int((input_box[2]-input_box[0])*.1)
        new_locations = np.array([max(input_box[0]-box_width,0), max(input_box[1]-box_height, 0), min(input_box[2]+box_width,w), min(input_box[3]+box_height,h)])
        new_cloth_img = cloth_img[new_locations[1]:new_locations[3],new_locations[0]:new_locations[2]]
        new_cloth_mask = binary_mask[new_locations[1]:new_locations[3],new_locations[0]:new_locations[2]]
        new_h, new_w, _ = new_cloth_img.shape
        original_aspect_ratio = new_w / new_h
        target_aspect_ratio = desired_width / desired_height
        if original_aspect_ratio > target_aspect_ratio:
            # Original is wider, resize based on width
            new_width = desired_width
            new_height = int(desired_width / original_aspect_ratio)
            resized_image = cv2.resize(new_cloth_img, (new_width, new_height))
            resized_mask = cv2.resize(new_cloth_mask, (new_width, new_height))
            # Calculate padding for top/bottom
            pad_vert = (desired_height - new_height) // 2
            padded_image = cv2.copyMakeBorder(resized_image, pad_vert, desired_height - new_height - pad_vert, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            padded_mask = cv2.copyMakeBorder(resized_mask, pad_vert, desired_height - new_height - pad_vert, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            # Original is taller, resize based on height
            new_height = desired_height
            new_width = int(desired_height * original_aspect_ratio)
            resized_image = cv2.resize(new_cloth_img, (new_width, new_height))
            resized_mask = cv2.resize(new_cloth_mask, (new_width, new_height))
            # Calculate padding for left/right
            pad_horiz = (desired_width - new_width) // 2
            padded_image = cv2.copyMakeBorder(resized_image, 0, 0, pad_horiz, desired_width - new_width - pad_horiz, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            padded_mask = cv2.copyMakeBorder(resized_mask, 0, 0, pad_horiz, desired_width - new_width - pad_horiz, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
        clothname_without_ext = os.path.splitext(os.path.basename(args.output_cloth_path))[0]
        cloth_name = f'./{clothname_without_ext}.jpg'
        cloth_mask_name = f'./{clothname_without_ext}_mask.jpg'
        padded_image = Image.fromarray(padded_image)
        padded_mask = Image.fromarray(padded_mask)
        padded_image.save(cloth_name)
        padded_mask.save(cloth_mask_name)
    

