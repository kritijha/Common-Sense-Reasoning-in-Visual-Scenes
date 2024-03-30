import numpy as np

import clip
import torch
from PIL import Image

import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

image = preprocess(Image.open("corgi.jpg")).unsqueeze(0).to(device)
print("Image shape ",image.shape)

text = clip.tokenize(["a dog"]).to(device)

with torch.no_grad():
    image_features,raw_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print("Image features shape : ",image_features.shape)
    print("Raw features shape : ",raw_features.shape)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
