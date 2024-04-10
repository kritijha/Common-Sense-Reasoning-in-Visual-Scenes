#pip install fiftyone
#pip install pytube

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np
import time
import torch.nn as nn
import math
import fiftyone.zoo as foz

# Convert video in path to array of frames at rate of 1 FPS
def process_video(path):
    KPS = 1 # Target Keyframes Per Second
    frame_array = []
    vidObj = cv2.VideoCapture(path)
    success = 1
    count = 0
    fps = round(vidObj.get(cv2.CAP_PROP_FPS))
    hop = round(fps / KPS)
    while(success):

        success,img = vidObj.read()

        if img is not None and count % hop == 0:
            resized = cv2.resize(img, (224, 224))

            im = Image.fromarray(np.uint8(resized)).convert('RGB')
            frame_array.append(im)

        count += 1

    return frame_array

# Load the pre-trained model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def extract_features(model,inputs,dummy_text_inputs):
    with torch.no_grad():
        outputs = model(input_ids=dummy_text_inputs, output_hidden_states=True, **inputs)
        penultimate_features = outputs.vision_model_output.last_hidden_state[:, :, :]  # [batch_size, sequence_length, hidden_size]
        return penultimate_features

class fastBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_layer = nn.AvgPool2d(3,stride=4)

    def forward(self,input):
        return self.pool_layer(input)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, drop_rate=0.1, max_len=3):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, inp):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        inp = inp + self.pe[:, :inp.size(1)]
        return inp

# Class to construct a dataset from a FiftyOne ActivityNet dataset based on params
class ActivitynetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name="activitynet-200", split="train", max_duration=15, max_samples=100):
        fiftyone_dataset = foz.load_zoo_dataset(dataset_name,
                                                split=split,
                                                max_duration=max_duration,
                                                max_samples=max_samples)
        self.samples = fiftyone_dataset.view()
        self.filepaths = self.samples.values('filepath')

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        sample = self.samples[filepath]
        frame_array = process_video(filepath)
        x = np.array(frame_array)

        # Get the input image size
        H = 16
        W = 16
        D = 1024
        final_vector = []
        total_number_of_frames = len(frame_array)
        fast_network = fastBranch()
        for frame_idx in range(total_number_of_frames):
            img = frame_array[frame_idx]

            img = processor(images=img,return_tensors="pt")
            height, width = img["pixel_values"].shape[2], img["pixel_values"].shape[3]

            # Create a batch of dummy text inputs
            batch_size = img["pixel_values"].size(0)
            dummy_text_inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=img["pixel_values"].device)
            features = extract_features(model,img,dummy_text_inputs)
            penultimate_features = features[:,1:,:]
            penultimate_features = penultimate_features.view(-1,H,W)

            low_resolution_features = fast_network(penultimate_features)
            feat = low_resolution_features.detach().numpy().flatten()
            final_vector.append(feat)

        DIMENSION = H*D
        final_vector_np = np.array(final_vector)

        positional_encoding = PositionalEncoding(DIMENSION,max_len=total_number_of_frames)
        pe_final_vector = positional_encoding(torch.tensor(final_vector_np))
        pe_final_vector = pe_final_vector.reshape((pe_final_vector.shape[0],16,1024,total_number_of_frames))
        y = pe_final_vector.reshape((pe_final_vector.shape[1],pe_final_vector.shape[2],pe_final_vector.shape[3]))

        # (T,224,224,3) and (16,1024, T)
        return x, y

    def __len__(self):
        return len(self.filepaths)

dataset = ActivitynetDataset("activitynet-200", max_duration=10, max_samples=100)

for data in dataset:
    x, y = data
    print(x.shape, y.shape)

