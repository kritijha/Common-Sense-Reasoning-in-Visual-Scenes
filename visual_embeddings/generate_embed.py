from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import cv2
import numpy as np
import time
import torch.nn as nn
import math
from matplotlib import cm
from slot_attention import SlotAttention


frame_array = []
video_path = '00000.mp4'

def process_video(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1

    while(success):

        success,img = vidObj.read()

        if img is not None:
            print("Type : ",type(img))
            #cv2.imwrite('../Frames/frame%d.jpg'% count,img)
            #img = img.reshape((img.shape[2],img.shape[1],img.shape[0]))
            #print("Img shape ",img.shape)
            int_opt = np.uint8(img)
            #print(int_opt.shape)
            im = Image.fromarray(np.uint8(img)).convert('RGB') 
            #print("Type after changing ",type(im))
            frame_array.append(im)
        count+=1

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

class fastBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_layer = nn.AvgPool2d(3,stride=4)
    
    def forward(self,input):
        return self.pool_layer(input)

# Load the pre-trained model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def extract_features(model,inputs,dummy_text_inputs):
    with torch.no_grad():
        outputs = model(input_ids=dummy_text_inputs, output_hidden_states=True, **inputs)
        penultimate_features = outputs.vision_model_output.last_hidden_state[:, :, :]  # [batch_size, sequence_length, hidden_size]
        return penultimate_features

# Prepare the input image
#image = Image.open("chair.jpg")
#inputs = processor(images=image, return_tensors="pt")

# Get the input image size
H = 16
W = 16
D = 1024

# Extract penultimate features


process_video(video_path)
final_vector = []
frame_np_array = np.array(frame_array)
#total_number_of_frames = len(frame_array)
total_number_of_frames = 3
fast_network = fastBranch()
time_start = time.time()
for frame_idx in range(total_number_of_frames):
    print("Frame_idx ",frame_idx)
    img = frame_array[frame_idx]
    
    #img.save('../Frames/frame%d.jpg'% frame_idx)
    img = processor(images=img,return_tensors="pt")
    height, width = img["pixel_values"].shape[2], img["pixel_values"].shape[3]

    # Create a batch of dummy text inputs
    batch_size = img["pixel_values"].size(0)
    dummy_text_inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=img["pixel_values"].device)
    #print("Device ",img["pixel_values"].device)
    features = extract_features(model,img,dummy_text_inputs)
    penultimate_features = features[:,1:,:]
    penultimate_features = penultimate_features.view(-1,H,W)

    print("Penultimate_shape ",penultimate_features.shape)
    low_resolution_features = fast_network(penultimate_features)
    print("Low resolution : ",low_resolution_features.shape)
    feat = low_resolution_features.detach().numpy().flatten()
    print("feat ",feat.shape)
    final_vector.append(feat)

print("Len of final vector ",len(final_vector))
print("Total number of frames ",total_number_of_frames)

final_vector_np = np.array(final_vector)
np.save('final_vec_np.npy',final_vector_np)
DIMENSION = H*D

print("Final vector shape ",final_vector_np.shape)

#low_resolution_features = fast_network(torch.tensor(final_vector_np))
#print(low_resolution_features)

positional_encoding = PositionalEncoding(DIMENSION,max_len=total_number_of_frames)
pe_final_vector = positional_encoding(torch.tensor(final_vector_np))
print("PE Final vector ",pe_final_vector.shape)

slot_attn = SlotAttention(
    num_slots = 3,
    dim = 16384,
    iters = 10
)

slots = slot_attn(pe_final_vector)
print(slots.shape)