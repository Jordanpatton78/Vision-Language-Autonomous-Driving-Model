import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Loop through directories of "../data/comma2k19/Chunk_1"
image_paths = []
speed_values = []
steering_values = []
discrete_speed_values = []
discrete_steering_values = []
data_dir = "../data/comma2k19/Chunk_1"
for outer_dir_name in os.listdir(data_dir):
    # Loop through inner directories
    outer_dir = os.path.join(data_dir, outer_dir_name)

    if os.path.isdir(outer_dir):
        for inner_dir_name in os.listdir(outer_dir):
            inner_dir = os.path.join(outer_dir, inner_dir_name)
            image = inner_dir + "/preview.png"
            speed_vals = np.load(inner_dir + "/processed_log/CAN/speed/value")
            steering_vals = np.load(inner_dir + "/processed_log/CAN/steering_angle/value")
            image_paths.append(image)
            speed_val = speed_vals[0][0]
            steering_val = steering_vals[0]
            speed_values.append(speed_val)
            steering_values.append(steering_val)
            if speed_val <= 2.5:
                discrete_speed = '0-5 mph'
            elif speed_val <= 15:
                discrete_speed = '5-35 mph'
            elif speed_val <= 25:
                discrete_speed = '35-55 mph'
            elif speed_val <= 30:
                discrete_speed = '55-65 mph'
            else:
                discrete_speed = '65+ mph'
            discrete_speed_values.append(discrete_speed)

            if steering_val <= -5:
                discrete_steering = 'Right'
            elif steering_val <= 5:
                discrete_steering = 'Straight'
            else:
                discrete_steering = 'Left'
            discrete_steering_values.append(discrete_steering)

# Create a DataFrame
data = {
    "image_path": image_paths,
    "speed": speed_values,
    "steering_angle": steering_values,
    "discrete_speed": discrete_speed_values,
    "discrete_steering": discrete_steering_values
}
df = pd.DataFrame(data)

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['discrete_steering'])
train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train['discrete_steering'])

# Save to CSV files
train.to_csv("../data/comma2k19_chunk1_train.csv", index=False)
val.to_csv("../data/comma2k19_chunk1_val.csv", index=False)
test.to_csv("../data/comma2k19_chunk1_test.csv", index=False)