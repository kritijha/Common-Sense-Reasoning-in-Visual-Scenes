from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load the pre-trained model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Prepare the input image
image = Image.open("chair.jpg")
inputs = processor(images=image, return_tensors="pt")

# Get the input image size
height, width = inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[3]

# Create a batch of dummy text inputs
batch_size = inputs["pixel_values"].size(0)
dummy_text_inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=inputs["pixel_values"].device)

# Extract penultimate features
with torch.no_grad():
    outputs = model(input_ids=dummy_text_inputs, output_hidden_states=True, **inputs)
    penultimate_features = outputs.vision_model_output.last_hidden_state[:, :, :]  # [batch_size, sequence_length, hidden_size]



# Remove the first token
penultimate_features = penultimate_features[:, 1:, :]  # [batch_size, 196, 1024]

# Reshape the tensor
penultimate_features = penultimate_features.view(penultimate_features.size(0), 16, 16, -1)

print(penultimate_features.shape)  # torch.Size([1, 16, 16, 1024])