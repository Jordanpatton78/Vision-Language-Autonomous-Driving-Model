# Use gemini 2.5 pro to take in images and generate rationale as to what a vehicle should do in this situation
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from PIL import Image
import dotenv
import base64
from tqdm.auto import tqdm
import textwrap
dotenv.load_dotenv("../../../../env")

API_KEY = os.getenv("OPENAI_API_KEY")

image_paths = []
speed_values = []
steering_values = []
discrete_speed_values = []
discrete_steering_values = []

client = OpenAI(api_key=API_KEY)

for data in ["train"]:
    df = pd.read_csv(f"../data/comma2k19_chunk1_{data}.csv")

    # df = df.iloc[5:6]

    rationale_list = []
    progress_bar = tqdm(total=len(df), desc="Generating rationales")
        
    for i in range(len(df)):
        image_path = df['image_path'].iloc[i]

        speed_target = df['discrete_speed'].iloc[i]
        steering_target = df['discrete_steering'].iloc[i]

        # Read local image bytes
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # Convert to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        input_text = textwrap.dedent(f"""
                        Analyze the driving scene in the provided image and produce a detailed rationale for the 
                        vehicle’s appropriate speed and steering decisions.

                        Your response must follow this structure, answering the questions below:

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

                        Output the prediction in this exact JSON format:

                        {{
                            "speed": "<one of the speed options>",
                            "steering": "<one of the steering options>"
                        }}

                        Hint: The vehicle’s actual speed is {speed_target}, and its steering value is {steering_target}. 
                        Use this only as subtle guidance to refine your rationale.
                        """)

        response = client.responses.create(
            model="o4-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": input_text},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                    ]
                }
            ]
        )

        rationale = response.output_text.strip()
        rationale_list.append(rationale)

        progress_bar.update(1)

    progress_bar.close()

    df['rationale'] = rationale_list
    df.to_csv(f"../data/comma2k19_chunk1_{data}_with_rationale.csv", index=False)