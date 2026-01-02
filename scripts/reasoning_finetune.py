from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import textwrap
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import torch.nn as nn
import re
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import peft
from peft import PeftModel
from PIL import Image
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import json
import random
import os
import numpy as np

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

train_data = pd.read_csv('../data/comma2k19_chunk1_train_with_rationale.csv')
val_data = pd.read_csv('../data/comma2k19_chunk1_val.csv')
test_data = pd.read_csv('../data/comma2k19_chunk1_test.csv')

lora_directory = "../models/best_reasoning_model"

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "../models/qwen3vl", dtype="auto", device_map="auto", local_files_only=True
)
processor = AutoProcessor.from_pretrained("../models/qwen3vl", local_files_only=True)

# Use LoRA for finetuneing
lora = peft.LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    task_type=peft.TaskType.CAUSAL_LM,
)
model = peft.get_peft_model(qwen_model, lora)

def reformat_rationale(text):
    """Wraps steps 1 & 2 in <think> and step 3 in <predict>."""
    # Pattern captures:
    # Group 1: Steps 1 & 2 content (from start or 1. to 3. Predicted Discrete Action)
    # Group 2: Step 3 content (from 3. Predicted Discrete Action to the end)
    pattern = re.compile(r'(.*)\n3\.\s*Predicted Discrete Action\s*(\{.*?\})', re.DOTALL | re.IGNORECASE)

    match = pattern.search(text)

    if match:
        # Step 1 & 2 content is captured in Group 1
        think_content = match.group(1).strip()
        # Step 3 content is the header and the JSON part, formatted slightly
        predict_content = f"3. Predicted Discrete Action \n{match.group(2).strip()}"

        return f"<think>\n{think_content}\n</think>\n\n<predict>\n{predict_content}\n</predict>"
    else:
        return text

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

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
num_epochs = 10
patience = 3
best_val_mc = float('-inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    for batch in train_dataloader:
        image_path = batch['image_path'][0]
        target_rationale = batch['rationale'][0]
        target_rationale = reformat_rationale(target_rationale)
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
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": target_rationale
                    }
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            conversation=conversation_with_response,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        input_messages_for_mask = [conversation_with_response[0]]

        input_tokens = processor.apply_chat_template(
            conversation=input_messages_for_mask,
            tokenize=True,
            add_generation_prompt=True, # Qwen uses an implicit prompt before the assistant's response
            return_dict=False,
            return_tensors="pt"
        )

        prompt_len = input_tokens.shape[1] + 1

        labels = inputs["input_ids"].clone()

        labels[:, :prompt_len] = -100

        inputs["labels"] = labels

        outputs = model(**inputs)

        # Compute loss here based on your target labels
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop can be added here with early stopping based on patience
    num_valid_speed_steering = 0
    val_speed_preds = []
    val_speed_targets = []
    val_steering_preds = []
    val_steering_targets = []
    for batch in val_dataloader:
        with torch.no_grad():
            image_path = batch['image_path'][0]
            target_speed = batch['discrete_speed'][0]
            target_steering = batch['discrete_steering'][0]
            image = Image.open(image_path).convert("RGB")

            messenges = [
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

            inputs = processor.apply_chat_template(
                conversation=messenges,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Extract predicted speed and steering from output_text
            output = output_text[0]

            json_pattern = re.compile(r"(?s)(\{.*?\})")

            json_group = json_pattern.search(output)

            if json_group:
                json_string = json_group.group(0)
                json_string = json_string.strip()

            else:
                print("Misformatted string extracted from output.")
                break

            pred_json = json.loads(json_string)

            predicted_speed_match = pred_json["speed"]

            predicted_steering_match = pred_json["steering"]

            # Only append if both predictions were successfully extracted
            if predicted_speed_match and predicted_steering_match:
                # Compare pred_speed and pred_steering with target_speed and target_steering
                val_speed_preds.append(predicted_speed_match)
                val_speed_targets.append(target_speed)
                val_steering_preds.append(predicted_steering_match)
                val_steering_targets.append(target_steering)
            else:
                # Optional: Log or print a warning if extraction fails to debug model output
                # print(f"Warning: Failed to extract both speed and steering for path: {batch['image_path'][0]}")
                # print(f"Model Output: {output}")
                # You might consider appending a default/null value if you want to prevent
                # the target from being ignored, but usually, skipping is safer for F1 score.
                print(f"Pred_speed and/ or pred_steering not found.")
    # Compute F1 score here based on your target discrete actions
    val_speed_mc = matthews_corrcoef(val_speed_targets, val_speed_preds)
    val_steering_mc = matthews_corrcoef(val_steering_targets, val_steering_preds)
    val_mc = (val_speed_mc + val_steering_mc) / 2
    if val_mc > best_val_mc:
        best_val_mc = val_mc
        epochs_without_improvement = 0
        model.save_pretrained(lora_directory)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

# Testing Loop
model = PeftModel.from_pretrained(
    qwen_model,
    lora_directory
)
test_speed_preds = []
test_speed_targets = []
test_steering_preds = []
test_steering_targets = []
for batch in test_dataloader:
        with torch.no_grad():
            image_path = batch['image_path'][0]
            target_speed = batch['discrete_speed'][0]
            target_steering = batch['discrete_steering'][0]
            image = Image.open(image_path).convert("RGB")

            messenges = [
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

            inputs = processor.apply_chat_template(
                conversation=messenges,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Extract predicted speed and steering from output_text
            output = output_text[0]

            json_pattern = re.compile(r"(?s)(\{.*?\})")

            json_group = json_pattern.search(output)

            if json_group:
                json_string = json_group.group(0)
                json_string = json_string.strip()

            else:
                print("Misformatted string extracted from output.")
                break

            pred_json = json.loads(json_string)

            predicted_speed_match = pred_json["speed"]

            predicted_steering_match = pred_json["steering"]

            print(f"\nImage path: {image_path}")

            print(f"\nOutput: {output_text}")

            print(f"\nCorrect discrete speed: {target_speed}")

            print(f"\nCorrect discrete steering: {target_steering}")

            print(f"\nPredicted discrete speed: {predicted_speed_match}")

            print(f"\nPredicted discrete steering: {predicted_steering_match}")

            # Only append if both predictions were successfully extracted
            if predicted_speed_match and predicted_steering_match:
                # Compare pred_speed and pred_steering with target_speed and target_steering
                test_speed_preds.append(predicted_speed_match)
                test_speed_targets.append(target_speed)
                test_steering_preds.append(predicted_steering_match)
                test_steering_targets.append(target_steering)
            else:
                # Optional: Log or print a warning if extraction fails to debug model output
                # print(f"Warning: Failed to extract both speed and steering for path: {batch['image_path'][0]}")
                # print(f"Model Output: {output}")
                # You might consider appending a default/null value if you want to prevent
                # the target from being ignored, but usually, skipping is safer for F1 score.
                print(f"Pred_speed and/ or pred_steering not found.")

# Compute F1 score here based on your target discrete actions
test_speed_f1 = f1_score(test_speed_targets, test_speed_preds, average='weighted')
test_steering_f1 = f1_score(test_steering_targets, test_steering_preds, average='weighted')
print(f"Test F1 Score for speed: {test_speed_f1}")
print(f"Test F1 Score for steering: {test_steering_f1}")
test_speed_mc = matthews_corrcoef(test_speed_targets, test_speed_preds)
test_steering_mc = matthews_corrcoef(test_steering_targets, test_steering_preds)
print(f"Test Matthew's Corr Coefficient for speed: {test_speed_mc}")
print(f"Test Matthew's Corr Coefficient for steering: {test_steering_mc}")
test_speed_acc = accuracy_score(test_speed_targets, test_speed_preds)
test_steering_acc = accuracy_score(test_steering_targets, test_steering_preds)
print(f"Test Accuracy for speed: {test_speed_acc}")
print(f"Test Accuracy for steering: {test_steering_acc}")


# Test F1 Score for speed: 0.48026315789473684
# Test F1 Score for steering: 0.718796992481203
# Test Matthew's Corr Coefficient for speed: 0.3789014749900143
# Test Matthew's Corr Coefficient for steering: 0.12980769230769232
# Test Accuracy for speed: 0.5
# Test Accuracy for steering: 0.7368421052631579
# Baseline speed: 0.356383
# Baseline steering: 0.787234