#!/bin/bash

# Accept positional parameters from the command line

video_path=$1
cloth_path=$2
body_option=$3
agnostic="./data/agnostic"
agnostic_mask="./data/agnostic_mask"
densepose="./data/densepose"
videos="./data/videos"
vivid_cloth="./data/cloth"
vivid_cloth_mask="./data/cloth_mask"
curr_dir=$(pwd)

timestamp=$(date +"%Y%m%d_%H%M%S")
videoname=$(basename -- "$video_path")
videoname_without_ext="${videoname%.*}"
extension="${videoname##*.}"
new_videoname="${videoname_without_ext}_${timestamp}.${extension}"
new_agnostic_videoname="${videoname_without_ext}_${timestamp}_agnostic.${extension}"
new_agnostic_mask_videoname="${videoname_without_ext}_${timestamp}_agnostic_mask.${extension}"
new_denseposename="${videoname_without_ext}_${timestamp}_densepose.${extension}"

clothname=$(basename -- "$cloth_path")
clothname_without_ext="${clothname%.*}"
cloth_extension="${clothname##*.}"
new_clothname="${clothname_without_ext}_${timestamp}.${cloth_extension}"
new_clothname_mask="${clothname_without_ext}_${timestamp}_mask.${cloth_extension}"

#cp "$video_path" "$curr_dir/$new_videoname"
python ./OOTDiffusion/run/resize_video.py --i "$video_path" --o "$curr_dir/$new_videoname"


if [[ "$body_option" == "full" ]]; then
    python ./OOTDiffusion/run/run_ootd_agnostic.py --model_path "$curr_dir/$new_videoname"  --cloth_path "$cloth_path" --output_cloth_path "$new_clothname" --scale 2.0 --sample 4 --category 2 --model_type dc --onnx "/mnt/data/vivid_ckpts/yolow-l.onnx" --text "cloth"
elif [[ "$body_option" == "upper" ]]; then 
    python ./OOTDiffusion/run/run_ootd_agnostic.py --model_path "$curr_dir/$new_videoname"  --cloth_path "$cloth_path" --output_cloth_path "$new_clothname" --scale 2.0 --sample 4 --category 0 --onnx "/mnt/data/vivid_ckpts/yolow-l.onnx" --text "cloth"
else
    echo "please use a correct mapping protocol! 'upper' or 'full'"
    exit 1
fi


cd "vid2densepose"

python main.py -i "$curr_dir/$new_videoname" -o "$curr_dir/$new_denseposename"

cd ..

if [[ -e "$curr_dir/$new_agnostic_videoname" && -e "$curr_dir/$new_agnostic_mask_videoname" && -e "$curr_dir/$new_agnostic_mask_videoname" && -e "$curr_dir/$new_denseposename" && -e "$curr_dir/$new_clothname" && -e "$curr_dir/$new_clothname_mask" ]]; then
    mv "$curr_dir/$new_videoname" "$videos"
    mv "$curr_dir/$new_agnostic_videoname" "$agnostic/$new_videoname"
    mv "$curr_dir/$new_agnostic_mask_videoname" "$agnostic_mask/$new_videoname"
    mv "$curr_dir/$new_denseposename" "$densepose/$new_videoname"
    mv "$curr_dir/$new_clothname" "$vivid_cloth/$new_clothname"
    mv "$curr_dir/$new_clothname_mask" "$vivid_cloth_mask/$new_clothname"
else
    echo "files missing, please check"
    exit 1
fi


python vivid.py --config "./configs/prompts/upper1.yaml" --video_path "$videos/$new_videoname" --cloth_path "$vivid_cloth/$new_clothname"
