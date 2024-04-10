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
import torch.nn.functional as F


video_path = '00000.mp4'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_f = 8 #number of fast slots.

def process_video(path):
    frame_array = []
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
            #print(int_opt.shape)
            im = Image.fromarray(np.uint8(img)).convert('RGB') 
            #print("Type after changing ",type(im))
            frame_array.append(im)
        count+=1
    return frame_array

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

class MLPDecoder(nn.Module):
    def __init__(self, hid_dim, out_dim, num_patches, num_slots):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = num_patches
        self.num_patches = num_patches
        self.num_slots = num_slots

        #self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, hid_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim)
        )

    def forward(self, x):
        # x is of shape [batch_size, num_slots, hid_dim]
        print("X shape ",x.shape)
        #x = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)  # Broadcast to number of patches
        #x = x + self.positional_encoding  # Add positional encoding
        print("X shape after repeat ",x.shape)

        y_hat = self.mlp(x)  # Apply MLP to each token
        alpha = F.softmax(y_hat, dim=1)  # Compute alpha map

        y = (y_hat * alpha).sum(dim=1)  # Weighted sum across slots

        return y
        
class Model(nn.Module):
    def __init__(self,pos_embed_dim,pos_max_len,total_number_of_frames,num_slots):
        super().__init__()
        self.pe = PositionalEncoding(pos_embed_dim,max_len=pos_max_len)
        self.slot_attn = SlotAttention(
            num_slots = N_f,
            dim = 1024,
            iters = 100
        )
        self.decoder = MLPDecoder(num_slots,16*1024,total_number_of_frames,num_slots)
    def forward(self,inputs):
        inputs = inputs.to(device)
        inputs = self.pe(inputs)
        print("Inputs shape ",inputs.shape)
        inputs = inputs.reshape((inputs.shape[0],16,total_number_of_frames,1024))
        slot_array = []
        for spatial_index in range(inputs.shape[1]):
            spatial_token = inputs[:,spatial_index,:,:]
            print("Spatial token shape ",spatial_token.shape)
            #spatial_token = spatial_token.reshape((spatial_token.shape[0],spatial_token.shape[2],spatial_token.shape[1]))
            print("Spatial token shape after reshape ",spatial_token.shape)
            slots = self.slot_attn(spatial_token)
            print("Slots shape ",slots.shape)
            slot_array.append(slots.cpu().detach().numpy())
        
        slot_array_tensor = np.array(slot_array)
        print("Slot array shape ",slot_array_tensor.shape)
        slot_array_tensor = slot_array_tensor.reshape((slot_array_tensor.shape[1],slot_array_tensor.shape[0]*slot_array_tensor.shape[3],slot_array_tensor.shape[2]))
        print("Slot array shape after reshape ",slot_array_tensor.shape)
        torch_slot_array = torch.tensor(slot_array_tensor)
        #torch_slot_array = torch_slot_array[:,:10,:]
        torch_slot_array = torch_slot_array.to(device)
        y = self.decoder(torch_slot_array)
        return y
    

class fastBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_layer = nn.AvgPool2d(3,stride=4)
    
    def forward(self,input):
        return self.pool_layer(input)
    
# Load the pre-trained model and processor
encoder_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
encoder_model.to(device)

def extract_features(model,inputs,dummy_text_inputs):
    inputs = inputs.to(device)
    dummy_text_inputs = dummy_text_inputs.to(device)
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
dataset_video_links = ['00000.mp4','00001.mp4','00002.mp4','00003.mp4','00004.mp4']
dataset_features = []
def create_dataset():
    for video_path in dataset_video_links:
        video_features = []
        frame_np_array = process_video(video_path)
        #total_number_of_frames = len(frame_np_array)
        total_number_of_frames = 20
        fast_network = fastBranch()
        time_start = time.time()
        for frame_idx in range(total_number_of_frames):
            print("Frame_idx ",frame_idx)
            img = frame_np_array[frame_idx]
            
            #img.save('../Frames/frame%d.jpg'% frame_idx)
            img = processor(images=img,return_tensors="pt")
            height, width = img["pixel_values"].shape[2], img["pixel_values"].shape[3]

            # Create a batch of dummy text inputs
            batch_size = img["pixel_values"].size(0)
            dummy_text_inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=img["pixel_values"].device)
            #print("Device ",img["pixel_values"].device)
            features = extract_features(encoder_model,img,dummy_text_inputs)
            penultimate_features = features[:,1:,:]
            penultimate_features = penultimate_features.view(-1,H,W)

            print("Penultimate_shape ",penultimate_features.shape)
            low_resolution_features = fast_network(penultimate_features)
            print("Low resolution : ",low_resolution_features.shape)
            feat = low_resolution_features.cpu().detach().numpy().flatten()
            print("feat ",feat.shape)
            video_features.append(feat)
        dataset_features.append(video_features)
    return np.array(dataset_features)
    

features = create_dataset()

print(features.shape)

total_number_of_frames = features.shape[1]

DIMENSION = H*D
N_f = 8
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# def __init__(self,pos_embed_dim,pos_max_len,total_number_of_frames,num_slots)
model = Model(DIMENSION,total_number_of_frames,total_number_of_frames,N_f)
model.to(device)
op = model(torch.tensor(features))
print("Op shape ",op.shape)
    


'''
print("Len of final vector ",len(final_vector))
print("Total number of frames ",total_number_of_frames)
 
final_vector_np = np.array(final_vector)
np.save('final_vec_np.npy',final_vector_np)
DIMENSION = H*D
N_f = 8

print("Final vector shape ",final_vector_np.shape)

#low_resolution_features = fast_network(torch.tensor(final_vector_np))
#print(low_resolution_features)

positional_encoding = PositionalEncoding(DIMENSION,max_len=total_number_of_frames)
pe_final_vector = positional_encoding(torch.tensor(final_vector_np))
pe_final_vector = pe_final_vector.reshape((pe_final_vector.shape[0],16,1024,total_number_of_frames))
print("PE Final vector ",pe_final_vector.shape)
pe_final_vector = pe_final_vector.reshape((pe_final_vector.shape[1],pe_final_vector.shape[2],pe_final_vector.shape[3]))

#pe_final_vector = pe_final_vector.reshape(pe_final_vector.shape[2],pe_final_vector.shape[1],pe_final_vector.shape[0])
print("After reshape ",pe_final_vector.shape)
slot_array = []

D = pe_final_vector[0].shape[0]
slot_attn = SlotAttention(
        num_slots = N_f,
        dim = D,
        iters = 100
    )
slot_attn.to(device)
print("Whole input shape ",pe_final_vector.shape)
pe_final_vector = pe_final_vector.reshape(pe_final_vector.shape[0],pe_final_vector.shape[2],pe_final_vector.shape[1])
pe_final_vector = pe_final_vector.to(device)
slots = slot_attn(pe_final_vector)
print("Optimized ",slots.shape)



# for spatial_token_id in range(pe_final_vector.shape[0]):
#     spatial_token = pe_final_vector[spatial_token_id]
#     #print("Spatial token shape ",spatial_token.shape)
#     T = spatial_token.shape[1]
#     D = spatial_token.shape[0]
#     #print("T ",T)
#     #print("D ",D)
#     spatial_token = spatial_token.reshape((1,T,D))
#     #print("Device : ",device)
#     spatial_token = spatial_token.to(device)
#     slots = slot_attn(spatial_token)
#     slots = slots.reshape((slots.shape[1],slots.shape[2]))
#     #print(slots.shape)
#     slot_array.append(slots.cpu().detach().numpy())

# slot_array = np.array(slot_array)
# print("Slot array shape! ",slot_array.shape)
'''