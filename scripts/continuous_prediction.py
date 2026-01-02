from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import textwrap
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import torch.nn as nn
import re
from sklearn.metrics import f1_score, accuracy_score
import peft
from peft import PeftModel
from PIL import Image
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import json
import numpy as np
import random
import time

# --- 1. Define the Seed Value ---
SEED = 42
print(f"Setting random seeds to: {SEED}")

# --- 2. Standard Python and OS Environment ---
random.seed(SEED)

# --- 3. NumPy ---
np.random.seed(SEED)

# --- 4. PyTorch (CPU and GPU) ---
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # --- 5. GPU Deterministic Flags (Crucial for reproducibility) ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_data = pd.read_csv("../data/comma2k19_chunk1_train_with_rationale.csv")
val_data = pd.read_csv("../data/comma2k19_chunk1_val.csv")
test_data = pd.read_csv("../data/comma2k19_chunk1_test.csv")

lora_directory = "../models/best_reasoning_model"

regression_head_path = "../models/best_reasoning_regression_head.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "../models/qwen3vl", dtype="auto", device_map="auto", local_files_only=True
)

lora_model = PeftModel.from_pretrained(
    qwen_model,
    lora_directory
)

processor = AutoProcessor.from_pretrained("../models/qwen3vl", local_files_only=True)

class RegressionHead(nn.Module):
    def __init__(self, lora_model, image_input_dim=1536, output_dim=2):
        super().__init__()
        self.lora_model = lora_model
        self.output_dim = output_dim
        self.hidden_size = self.lora_model.config.text_config.hidden_size + image_input_dim
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )

    def forward(self, **inputs):
        outputs = self.lora_model(**inputs, output_hidden_states=True)

        attention_mask = inputs['attention_mask']

        last_hidden_state = outputs.hidden_states[-1]

        sequence_lengths = attention_mask.sum(dim=1) - 1

        last_token_states = last_hidden_state[
            torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
            sequence_lengths
        ]

        image_inputs = inputs['image_inputs'].to(last_hidden_state.dtype)

        regression_inputs = torch.cat([last_token_states, image_inputs], dim=1)

        regression_output = self.regression_head(regression_inputs)

        return regression_output

model = RegressionHead(lora_model).to(device)
model = model.to(dtype=torch.bfloat16)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

train_dl = DataLoader(train_dataset, batch_size=1)
val_dl = DataLoader(val_dataset, batch_size=1)
test_dl = DataLoader(test_dataset, batch_size=1)

optim = AdamW(model.parameters(), lr=1e-3)
epochs = 20
patience = 3
epochs_wo_improve = 0
loss_fn = nn.MSELoss()
best_val_mse = float("inf")

input_text = textwrap.dedent("""Analyze the driving scene in the provided image and produce a detailed rationale for the
        vehicle’s appropriate speed and steering decisions, before producing final discrete predictions.

        Your response must follow this structure, answering the questions below, with the rationale contained in <think>...</think> tags:

        1. Speed
        - Where is the vehicle driving (freeway, city streets, etc.)? What is a typical speed in this environment?
        - What are the driving conditions (clear, snowing, raining, etc.)? How should this impact the speed the vehicle is moving?
        - What other agents are in the scene (other vehicles, bikers, pedestrians, etc.)? What are their potential actions (slowing down, walking in front of the car, swerving into the lane, etc)? How should that impact the driver's speed?

        2. Steering Angle
        - What is the curvature of the lane the vehicle is driving in? Is the lane straight, curved left, curved right, or other?
        - Are there any potential obstacles in the drivers way? If so, how should the driver steer to avoid these obstacles?
        - What other agents are in the scene (other vehicles, bikers, pedestrians, etc.)? What are their potential actions (slowing down, walking in front of the car, swerving into the lane, etc)? How should that impact the driver's steering angle?

        3. Predicted Discrete Action
        Provide your final discrete predictions using ONLY the allowed values:

        Speed options: "0-5 mph", "5-35 mph", "35-55 mph", "55-65 mph", "65+ mph"
        Steering options: "Left", "Straight", "Right"
        Note that "Left" and "Right" can be minimal adjustments, not just sharp turns. If the road curves slightly, "Left" or "Right" is appropriate.

        Now, analyze the image and produce the complete response, for example:

        <think>
        1. Speed  
        - Where is the vehicle driving? What is a typical speed in this environment?  
        The vehicle is on a multi-lane freeway at night. Typical freeway speeds in this setting range from 55 to 75 mph, depending on posted limits and traffic flow.  
        - What are the driving conditions? How should this impact the speed?  
        The pavement appears dry, visibility is reduced due to darkness but there is no rain or snow. Under clear but low-light conditions, the driver should maintain freeway speed while remaining vigilant, possibly at the upper end of the speed limit if comfortable and conditions remain stable.  
        - What other agents are in the scene? What are their potential actions? How should that impact the driver’s speed?  
        There are very few vehicles visible: one car in an adjacent lane ahead to the right, and tail lights in the distance. No pedestrians or cyclists. Other vehicles could brake or change lanes unexpectedly, so the driver should keep a safe following distance and be ready to reduce speed, but can otherwise maintain a steady freeway pace.

        2. Steering Angle  
        - What is the curvature of the lane?  
        The lane is essentially straight with only a very gentle alignment; no significant left or right bend is visible.  
        - Are there any potential obstacles in the driver’s way? If so, how should the driver steer to avoid them?  
        There are no obstacles on the roadway surface. Guardrails and signage are off the travel lanes. No steering adjustments are needed beyond maintaining lane centering.  
        - What other agents are in the scene? What are their potential actions? How should that impact the driver’s steering angle?  
        The lone vehicle to the right could drift or change lanes; the driver should maintain lane discipline and be prepared to make a slight corrective steering input if another car encroaches, but otherwise hold a straight steering angle.
        </think>

        <predict>
        3. Predicted Discrete Action  

        {
            "speed": "65+ mph",
            "steering": "Straight"
        }
        </predict>
        """)

for epoch in range(epochs):
    model.train()
    for batch in train_dl:
        image_path = batch['image_path'][0]
        target_speed = batch['speed'][0]
        target_steering = batch['steering_angle'][0]
        image = Image.open(image_path).convert("RGB")

        conversation_with_response = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": input_text
                    }
                ]
            }
        ]

        image_inputs = processor(
            images=image, 
            text=input_text,
            return_tensors="pt"
        ).to(device)

        feature_sequence = image_inputs['pixel_values'].to(device)

        pooled_image_features = feature_sequence.mean(dim=0).unsqueeze(0)

        inputs = processor.apply_chat_template(
            conversation=conversation_with_response,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )

        inputs['image_inputs'] = pooled_image_features
        inputs = inputs.to(device)

        outputs = model(**inputs)

        speed_preds = outputs[:, 0:1]

        steering_preds = outputs[:, 1:2]

        targets = torch.tensor([[target_speed, target_steering]], dtype=torch.float32).to(device)

        loss = loss_fn(outputs.float(), targets)

        loss.backward()

        optim.step()

        optim.zero_grad()

    model.eval()
    val_speed_preds = []
    val_steering_preds = []
    val_speed_targets = []
    val_steering_targets = []
    for batch in val_dl:
        with torch.no_grad():
            image_path = batch['image_path'][0]
            target_speed = batch['speed'][0]
            target_steering = batch['steering_angle'][0]
            image = Image.open(image_path).convert("RGB")

            conversation_with_response = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": input_text
                        }
                    ]
                }
            ]

            image_inputs = processor(
                images=image, 
                text=input_text,
                return_tensors="pt"
            ).to(device)

            feature_sequence = image_inputs['pixel_values'].to(device)

            pooled_image_features = feature_sequence.mean(dim=0).unsqueeze(0)

            inputs = processor.apply_chat_template(
                conversation=conversation_with_response,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt"
            )

            inputs['image_inputs'] = pooled_image_features
            inputs = inputs.to(device)

            outputs = model(**inputs)

            speed_preds = outputs[:, 0:1]

            steering_preds = outputs[:, 1:2]

            val_speed_preds.append(speed_preds)

            val_steering_preds.append(steering_preds)

            val_speed_targets.append(target_speed)

            val_steering_targets.append(target_steering)

    val_speed_preds = torch.cat(val_speed_preds).cpu()

    val_speed_targets = torch.tensor(val_speed_targets, dtype=torch.float32).unsqueeze(1)

    val_steering_targets = torch.tensor(val_steering_targets, dtype=torch.float32).unsqueeze(1)

    val_steering_preds = torch.cat(val_steering_preds).cpu()

    val_speed_mse = loss_fn(val_speed_preds, val_speed_targets)

    val_steering_mse = loss_fn(val_steering_preds, val_steering_targets)

    avg_mse = (val_speed_mse + val_steering_mse)/ 2

    if avg_mse < best_val_mse:
        best_val_mse = avg_mse
        torch.save(model.state_dict(), regression_head_path)
        epochs_wo_improve = 0
    else:
        epochs_wo_improve += 1

    if epochs_wo_improve >= patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "../models/qwen3vl", dtype="auto", device_map="auto", local_files_only=True
)

lora_model = PeftModel.from_pretrained(
    qwen_model,
    lora_directory
)

model = RegressionHead(lora_model).to(device)
model.load_state_dict(torch.load(regression_head_path))
model = model.to(dtype=torch.bfloat16)

model.eval()
test_speed_preds = []
test_steering_preds = []
test_speed_targets = []
test_steering_targets = []
start_time = time.perf_counter()
for batch in test_dl:
    with torch.no_grad():
        image_path = batch['image_path'][0]
        target_speed = batch['speed'][0]
        target_steering = batch['steering_angle'][0]
        image = Image.open(image_path).convert("RGB")

        conversation_with_response = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": input_text
                    }
                ]
            }
        ]

        image_inputs = processor(
            images=image, 
            text=input_text,
            return_tensors="pt"
        ).to(device)

        feature_sequence = image_inputs['pixel_values'].to(device)

        pooled_image_features = feature_sequence.mean(dim=0).unsqueeze(0)

        inputs = processor.apply_chat_template(
            conversation=conversation_with_response,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )

        inputs['image_inputs'] = pooled_image_features
        inputs = inputs.to(device)

        outputs = model(**inputs)

        speed_preds = outputs[:, 0:1]

        steering_preds = outputs[:, 1:2]

        test_speed_preds.append(speed_preds)

        test_steering_preds.append(steering_preds)

        test_speed_targets.append(target_speed)

        test_steering_targets.append(target_steering)

end_time = time.perf_counter()

duration = end_time - start_time

avg_dur = duration/ len(test_dl)

test_speed_preds = torch.cat(test_speed_preds).cpu()

test_speed_targets = torch.tensor(test_speed_targets, dtype=torch.float32).unsqueeze(1)

test_steering_targets = torch.tensor(test_steering_targets, dtype=torch.float32).unsqueeze(1)

test_steering_preds = torch.cat(test_steering_preds).cpu()

test_speed_mse = loss_fn(test_speed_preds, test_speed_targets)

test_steering_mse = loss_fn(test_steering_preds, test_steering_targets)

print(f"Average speed RMSE: {np.sqrt(test_speed_mse)}")
print(f"Average steering RMSE: {np.sqrt(test_steering_mse)}")
print(f"Average latency per prediction: {avg_dur}")

# RMSE speed using just last token: 8.075033187866211
# RMSE steering using just last token: 8.722293853759766

# RMSE speed using last token + image: 3.264098882675171
# RMSE steering using last token + image: 6.014684200286865
# Average latency per prediction: 0.12727377402571668 seconds