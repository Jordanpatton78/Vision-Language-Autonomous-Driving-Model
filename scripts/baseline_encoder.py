from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import torch.nn as nn
import pandas as pd
import os
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from huggingface_hub import snapshot_download

# --- Setup ---
torch.manual_seed(1234)
IMAGE_ROOT = '../data/' 

train_data = pd.read_csv('../data/comma2k19_chunk1_train.csv')
val_data = pd.read_csv('../data/comma2k19_chunk1_val.csv')
test_data = pd.read_csv('../data/comma2k19_chunk1_test.csv')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# --- Model and Processor Loading ---
MODEL_NAME = "../models/qwen3vl"
processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
# NOTE: We only load the model here to check the configuration, but we won't use its forward pass.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, dtype="auto", device_map="auto", local_files_only=True
)

input_text = """Analyze the driving scene in the provided image and produce a detailed rationale for the
        vehicle’s appropriate speed and steering decisions, before producing final discrete predictions.

        Your response must follow this structure, with the rationale contained in <think>...</think> tags:

        1. Scene Description
        - Describe the road geometry, number of lanes, curvature, and visibility.
        - Describe weather, lighting, surface conditions, and any relevant markings.
        - Describe all traffic participants (vehicles, pedestrians, cyclists) and their
        relative positions and motion.
        - Describe traffic controls (lights, signs, signals) and any potential hazards
        or emerging risks.

        2. Action Rationale
        - Explain, step by step, what the vehicle should do and why.
        - Reference the visual cues that justify the recommended behavior
        (e.g., distance to lead vehicle, lane boundaries, road curvature, traffic rules).
        - Ensure the rationale is grounded in safe autonomous-driving practices.

        3. Predicted Discrete Action
        Provide your final discrete predictions using ONLY the allowed values:

        Speed options: "Stopped", "Slow", "Medium", "Fast", "Very Fast"
        Steering options: "Left", "Straight", "Right"
        Note that "Left" and "Right" can be minimal adjustments, not just sharp turns. If the road curves slightly, "Left" or "Right" is appropriate.

        Following the </think> tag, output the prediction in this exact JSON format:

        {
        "speed": "<one of the speed options>",
        "steering": "<one of the steering options>"
        }
        """

# 1. Correct Dimension: Use the feature dimension found in the traceback.
# The sequence length is 3888, the feature dimension is 1536.
IMAGE_EMBEDDING_DIM = 1536 

# --- Regression Head Definition (Updated Dimension) ---
class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.regressor(x)

regression_head = RegressionHead(IMAGE_EMBEDDING_DIM).to(device)

train_ds = Dataset.from_pandas(train_data)
train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in ['image_path', 'speed', 'steering_angle']])
train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)

val_ds = Dataset.from_pandas(val_data)
val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in ['image_path', 'speed', 'steering_angle']])
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

test_ds = Dataset.from_pandas(test_data)
test_ds = test_ds.remove_columns([col for col in test_ds.column_names if col not in ['image_path', 'speed', 'steering_angle']])
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(regression_head.parameters(), lr=1e-4)

num_epochs = 10
patience = 3
best_loss = float('inf')
epochs_no_improve = 0
train_losses = []
for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):

        image_filename = batch['image_path'][0]
        image_path = os.path.join(IMAGE_ROOT, image_filename)
        image = Image.open(image_path).convert("RGB")

        target_speed = float(batch['speed'][0])
        target_steering = float(batch['steering_angle'][0])

        # Pass the image and an empty text string to the processor
        inputs = processor(
            images=image, 
            text=input_text,
            return_tensors="pt"
        ).to(device)
        
        # Extract the pre-encoded feature sequence (SequenceLength, FeatureDimension)
        feature_sequence = inputs['pixel_values'].to(device)
        
        # Shape check (Optional, for debugging):
        # print(f"Encoded Feature Shape: {feature_sequence.shape}") 
        
        # 3. Apply Global Average Pooling (mean over sequence dimension 0)
        # The shape is (SequenceLength, FeatureDimension) in this output, so pool over dim 0.
        pooled_features = feature_sequence.mean(dim=0).unsqueeze(0) # Unsqueeze to get (1, FeatureDimension)
        
        # 4. Pass the pooled features through the Regression Head
        outputs = regression_head(pooled_features)

        loss = loss_fn(outputs, torch.tensor([[target_speed, target_steering]], device=device, dtype=torch.float))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    val_losses = []
    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            image_filename = batch['image_path'][0]
            image_path = os.path.join(IMAGE_ROOT, image_filename)
            image = Image.open(image_path).convert("RGB")

            target_speed = float(batch['speed'][0])
            target_steering = float(batch['steering_angle'][0])

            # Pass the image and an empty text string to the processor
            inputs = processor(
                images=image, 
                text=input_text,
                return_tensors="pt"
            ).to(device)
            
            # Extract the pre-encoded feature sequence (SequenceLength, FeatureDimension)
            feature_sequence = inputs['pixel_values'].to(device)
            
            # Shape check (Optional, for debugging):
            # print(f"Encoded Feature Shape: {feature_sequence.shape}") 
            
            # 3. Apply Global Average Pooling (mean over sequence dimension 0)
            # The shape is (SequenceLength, FeatureDimension) in this output, so pool over dim 0.
            pooled_features = feature_sequence.mean(dim=0).unsqueeze(0) # Unsqueeze to get (1, FeatureDimension)
            
            # 4. Pass the pooled features through the Regression Head
            outputs = regression_head(pooled_features)

            loss = loss_fn(outputs, torch.tensor([[target_speed, target_steering]], device=device, dtype=torch.float))
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(regression_head.state_dict(), '../models/best_regression_head.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Testing Loop
regression_head.load_state_dict(torch.load('../models/best_regression_head.pth'))
test_speed_losses = []
test_steering_losses = []
for batch in tqdm(test_dataloader):
    with torch.no_grad():
        image_filename = batch['image_path'][0]
        image_path = os.path.join(IMAGE_ROOT, image_filename)
        image = Image.open(image_path).convert("RGB")

        target_speed = float(batch['speed'][0])
        target_steering = float(batch['steering_angle'][0])

        # Pass the image and an empty text string to the processor
        inputs = processor(
            images=image, 
            text=input_text,
            return_tensors="pt"
        ).to(device)
        
        # Extract the pre-encoded feature sequence (SequenceLength, FeatureDimension)
        feature_sequence = inputs['pixel_values'].to(device)
        
        # Shape check (Optional, for debugging):
        # print(f"Encoded Feature Shape: {feature_sequence.shape}") 
        
        # 3. Apply Global Average Pooling (mean over sequence dimension 0)
        # The shape is (SequenceLength, FeatureDimension) in this output, so pool over dim 0.
        pooled_features = feature_sequence.mean(dim=0).unsqueeze(0) # Unsqueeze to get (1, FeatureDimension)
        
        # 4. Pass the pooled features through the Regression Head
        outputs = regression_head(pooled_features)

        test_speed_losses.append((loss_fn(outputs[:, 0], torch.tensor([target_speed], device=device, dtype=torch.float))).item())
        test_steering_losses.append((loss_fn(outputs[:, 1], torch.tensor([target_steering], device=device, dtype=torch.float))).item())
avg_speed_loss = sum(test_speed_losses) / len(test_speed_losses)
avg_steering_loss = sum(test_steering_losses) / len(test_steering_losses)
print(f"Average speed RMSE: {np.sqrt(avg_speed_loss)}")
print(f"Average steering RMSE: {np.sqrt(avg_steering_loss)}")

# Average speed RMSE: 11.758290876251092
# Average steering RMSE: 3.4734676105635525