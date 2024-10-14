import os
from pathlib import Path
import sys
from PIL import Image
from utils_ootd import get_mask_location
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

import json
import argparse
import os.path as osp
import supervision as sv
import onnxruntime as ort

try:
    import torch
    from torchvision.ops import nms
except Exception as e:
    print(e)

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))


BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser('YOLO-World ONNX Demo')
    parser.add_argument('--onnx', default="", help='onnx file', required=False)
    parser.add_argument('--image', default="", help='image path, include image file or dir.', required=False)
    parser.add_argument(
        '--text', default="",
        help=
        'detecting texts (str or json), should be consistent with the ONNX model', required=False
    )
    parser.add_argument('--output-dir',
                        default='./',
                        help='directory to save output files', required=False)
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference', required=False)
    parser.add_argument(
        '--onnx-nms',
        default=True,
        action='store_false',
        help='whether ONNX model contains NMS and postprocessing', required=False)
    args, unknown = parser.parse_known_args()
    args.onnx = "/mnt/data/vivid_ckpts/yolow-l.onnx"
    args.image = "/home/zenan/ViViD/data/cloth/1055822_in_xl.jpg"
    args.text = "cloth"

    return args


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


def visualize(image, bboxes, labels, scores, texts):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def inference(ort_session,
              image_path,
              texts,
              output_dir,
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
    """
    bboxes = bboxes[0:1]
    bboxes /= scale_factor
    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
    bboxes = bboxes.round().astype('int')
    #image *= 255
    #image = image.astype(np.uint8)[0]
    image_out = visualize(ori_image, bboxes, labels[0:1], scores[0:1], texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)
    return image_out
    """

def inference_with_postprocessing(ort_session,
                                  image_path,
                                  texts,
                                  output_dir,
                                  size=(640, 640),
                                  nms_thr=0.7,
                                  score_thr=0.3,
                                  max_dets=300):
    # export with `--without-nms`
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["scores", "boxes"], {"images": input_ort})
    scores, bboxes = results
    # move numpy array to torch
    ori_scores = torch.from_numpy(scores[0]).to('cuda:0')
    ori_bboxes = torch.from_numpy(bboxes[0]).to('cuda:0')

    scores_list = []
    labels_list = []
    bboxes_list = []
    # class-specific NMS
    for cls_id in range(len(texts)):
        cls_scores = ori_scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = nms(ori_bboxes, cls_scores, iou_threshold=nms_thr)
        cur_bboxes = ori_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]
    if len(keep_idxs) > max_dets:
        _, sorted_idx = torch.sort(scores, descending=True)
        keep_idxs = sorted_idx[:max_dets]
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]

    # Get candidate predict info by num_dets
    scores = scores.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
    bboxes = bboxes.round().astype('int')

    image_out = visualize(ori_image, bboxes, labels, scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)
    return image_out


def main():

    args = parse_args()
    onnx_file = args.onnx
    # init ONNX session
    ort_session = ort.InferenceSession(
        onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("Init ONNX Runtime session")
    output_dir = "onnx_outputs"
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    sam = sam_model_registry['vit_h'](checkpoint="/mnt/data/vivid_ckpts/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    
    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

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
    else:
        inference_func = inference_with_postprocessing

    for img in images:
        input_box = inference_func(ort_session, img, texts, output_dir=output_dir)
        
        cloth_path = img
        cloth_img = np.array(Image.open(cloth_path))
        h, w, _ = cloth_img.shape

        predictor.set_image(cloth_img)
        masks, _, _ = predictor.predict(box=input_box,multimask_output=False)
        binary_mask = masks[0].astype(np.uint8) * 255
        binary_mask = cv2.cvtColor(binary_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite("orig_mask.jpg",binary_mask)
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

        padded_image = Image.fromarray(padded_image)
        padded_mask = Image.fromarray(padded_mask)
        padded_image.save('./resized_img.jpg')
        padded_mask.save('./resized_mask.jpg')
        

    print("Finish inference")


if __name__ == "__main__":
    main()

"""
cloth_path = args.cloth_path
sam = sam_model_registry['vit_h'](checkpoint="/mnt/data/vivid_ckpts/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

if __name__ == '__main__':
    #count = 0
    cloth_img = np.array(Image.open(cloth_path))
"""

    

