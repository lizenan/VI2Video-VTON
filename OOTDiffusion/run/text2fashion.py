import requests
import re
import os
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
#from openai import OpenAI
import time
import clip
from PIL import Image
import argparse
import shutil
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
from datetime import datetime

# Initialize OpenAI API

client = OpenAI(api_key = '<api-key>')

parser = argparse.ArgumentParser(description='text2fashion')
parser.add_argument('--images_path', type=str, default="", required=True)
parser.add_argument('--text_description', type=str, default="", required=True)
parser.add_argument('--generate', type=int, default=False, required=False)
parser.add_argument('--o', type=str, default="", required=True)
args = parser.parse_args()

images_path = args.images_path
text = args.text_description
output_path = args.o
retrieval_flag = args.generate

if not retrieval_flag:  #retrieval method
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    def browse_media_files(directory):
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            return
        
        image_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                _, extension = os.path.splitext(file)
                extension = extension.lower()
                if extension in IMAGE_EXTENSIONS:
                    image_files.append(file)

        return image_files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-L/14", device=device)

    #def preprocess_image(image_path):
    #    image = Image.open(image_path)
    #    return preprocess(image).unsqueeze(0).to(device)

    class ImageDataset(Dataset):
        def __init__(self, image_dir, image_files, transform=None):
            self.image_dir = image_dir
            self.image_files = image_files
            self.transform = transform

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")  # Convert image to RGB mode

            if self.transform:
                image = self.transform(image)
            
            return image, self.image_files[idx]
        


    if not os.path.exists(os.path.join(images_path, "fashion_embeddings.npy")):
        print("preparing embedding files")
        images_files = browse_media_files(images_path)
        #all_images_files_batched = np.array_split(images_files, 200)
        dataset = ImageDataset(image_dir=images_path, image_files=images_files, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=32)
        img_embeddings = []
        img_filenames = []
        for images_file_batch in tqdm(dataloader):
            image_tensors,batch_filenames = images_file_batch#torch.cat([preprocess_image(os.path.join(images_path,image_path)) for image_path in images_file_batch])
            image_tensors = image_tensors.to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_embeddings.append(image_features.cpu())
            img_filenames.append(list(batch_filenames))

        img_embeddings = torch.cat(img_embeddings).numpy()
        all_img_filenames = np.concatenate(img_filenames)
        np.save(os.path.join(images_path,"fashion_embeddings.npy"), img_embeddings)
        np.save(os.path.join(images_path,"fashion_filenames.npy"), all_img_filenames)
    else:
        all_img_filenames = np.load(os.path.join(images_path,"fashion_filenames.npy"))
        img_embeddings = np.load(os.path.join(images_path,"fashion_embeddings.npy"))



    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    img_embeddings = torch.from_numpy(img_embeddings).to(text_features.device)

    topk = 1
    text_sim = text_features @ img_embeddings.T
    index_topk = text_sim.argsort()[0,-topk:]
    selected_filename = all_img_filenames[index_topk]
    shutil.copy(os.path.join(images_path,selected_filename), os.path.join(output_path,selected_filename))
    selected_filename = os.path.join(images_path,selected_filename)
else:
    response = client.images.generate(
    model="dall-e-3",
    prompt=text,
    size="1024x1024",
    quality="standard",
    n=1,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_filename = os.path.join(images_path,f"{timestamp}.jpg")
    image_url = response.data[0].url

    image_response = requests.get(image_url, stream=True)

    # Check if the request was successful
    if image_response.status_code == 200:
        # Open a local file for writing the downloaded image
        with open(selected_filename, 'wb') as file:
            # Write the content in chunks to avoid loading the entire image into memory at once
            for chunk in image_response.iter_content(1024):
                file.write(chunk)



#shutil.copy(os.path.join(images_path,selected_filename), os.path.join(output_path,selected_filename))
try:
    written_file_path = os.path.join(output_path,'selected_cloth.txt')
    with open(written_file_path, 'w') as file:
        file.write(selected_filename)
    print(f"Information written to '{written_file_path}'")
except Exception as e:
    print(f"Error writing to file: {e}")

