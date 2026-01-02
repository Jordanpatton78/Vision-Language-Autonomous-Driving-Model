Project Report: End-to-End Self-Driving with Vision-Language Models
1. Project Overview
The self-driving car company Waymo recently wrote a paper about their desired approach to end-to-end self-driving using what they call a “foundation model for self-driving” called EMMA. This project is inspired by EMMA, adjusted to fit in the 30 hour time constraint. 
The goal is simple: train a vision-language model (VLM) to be able to reason effectively about driving scenarios and make accurate continuous predictions of vehicle speed and steering angle in real-world driving scenarios.

2. Dataset Description
2.1 Dataset Source
I used the dataset Comma2k19 from Comma.ai. It is a dataset of videos taken while driving on a California freeway, with accompanying vehicle speed and steering angle measurements, which act as my targets. For simplicity, I only used preview images as the inputs into my model instead of full videos.

3. Problem Formulation
3.1 Type of Problem
This is primarily a regression problem with two objectives:
Minimize mean squared error (MSE) for predicted vehicle speed
Minimize mean squared error (MSE) for predicted vehicle steering angle
I also evaluate the reasoning capabilities of my model by evaluating how well my model can make predictions in its chain-of-thought reasoning by predicting within the following discrete classes:
Vehicle speed is discretized into the following bins:
0-5 mph
5-35 mph
35-55 mph
55-65 mph
65+ mph
Steering angle is discretized into the following bins:
Right (<-5 degrees)
Left (>5 degrees)
Straight (>-5 degrees and <5 degrees)
I’m evaluating chain-of-thought reasoning classification performance with the following metrics:
Exact accuracy
F1 Score
Matthew’s Correlation Coefficient (Useful for unbalanced datasets)
Another important factor I paid attention to, although it was not a training objective, is latency. A real-world self-driving car would need to produce accurate predictions very quickly, so even if a model performs well, it would be insufficient if it is too slow.
3.2 Learning Paradigm
This is a supervised learning task since target vehicle speeds and steering angles are available.
3.3 Background Knowledge
I have experience with both computer vision and natural language processing problems that were useful for this project. I have previously built a lane-detection deep learning model and have worked with reasoning models in my job.
3.4 Related Work
The EMMA paper describes state-of-the-art behavior in many end-to-end self-driving tasks. My approach of predicting continuous vehicle speed and steering angle using a reasoning-focused VLM is novel, so my results are unique.

4. Dataset Exploration
Before starting implementation, I explored the dataset to understand its structure and distributions.
4.1 Observations
My primary observations concern the uneven distributions of the target variables. Since this data comes primarily from freeway driving, there is a skew toward faster speeds, with over 55% of the recorded vehicle speeds exceeding 55 mph.
There is an even greater disparity with steering angle. The vast majority of the data (>78%) has a steering angle between -5 and 5 degrees (0 being perfectly straight), which corresponds to relatively straight driving. As a result, most of the driving data represents going straight.
4.2 Visualizations

	

Example preview image of vehicle driving straight between 0-5 mph.
Distribution of speeds, discretized into bins. Most of the speeds (>55%) are >55 mph. 
Distribution of steering angles, discretized into bins. The vast majority (>78%) are near 0, indicating the driver was driving straight ahead.
KDE plot of speeds, which reveals that there are several instances of driving near 0 mph.
KDE plot of steering angles, which reinforces that the vast majority of the time the driving angle is near 0.

5. Methodology
5.1 Overall Approach
My approach consists of the following steps:
Establish baseline performance by adding a regression head to the vision encoder only.
Train the full VLM to produce accurate chain-of-thought reasoning and evaluate its performance on discrete prediction targets identified during EDA.
Add a regression head to the final hidden layer of the VLM to predict continuous vehicle speed and steering angle.
Train a regression head that uses both final token embeddings and image information.
Distill the knowledge from the reasoning model into a smaller, vision-only model that learns by matching embeddings, with the goal of significantly reducing inference latency.

6. Model Architecture
I used Qwen 3 VL 2B as the base VLM. When fine-tuning the model to produce improved reasoning, I added LoRA adapters and trained those parameters. I also added various regression heads to produce continuous outputs.
An AdamW optimizer was used during training.

7. Training and Inference Procedure
7.1 Reasoning Model Training
To train the reasoning model, I used an approach adapted from the STaR paper to generate high-quality rationales. I used another VLM (GPT-4o) to generate rationales that led to correct discrete predictions of speed and steering angle.
I then fine-tuned the Qwen model to produce the same rationales before making its own discrete predictions of speed and steering angle.
7.2 Continuous Prediction Models
For continuous prediction, I took the fine-tuned Qwen model and added a regression head on the embeddings before the final token prediction. The model outputs two values: predicted speed and predicted steering angle. Training minimized the MSE between predicted values and the true speed and steering angle.

8. Data Splitting Strategy
I randomly sampled 20% of the image–speed–steering pairs for the test set. From the remaining data, I sampled 10% to form a validation set. All experiments used the same training, validation, and test splits.

9. Results
9.1 Baseline Vision Encoder Regression
Speed RMSE (mph): 11.758290876251092
Steering RMSE (degrees): 3.4734676105635525
9.2 Chain-of-Thought Discrete Performance
Test Accuracy (speed): 0.5
Test Accuracy (steering): 0.7368421052631579
Test F1 Score (speed): 0.48026315789473684
Test F1 Score (steering): 0.718796992481203
Test Matthews Correlation Coefficient (speed): 0.3789014749900143
Test Matthews Correlation Coefficient (steering): 0.12980769230769232
9.3 Reasoning Regression Model
Speed RMSE (mph): 8.075033187866211
Steering RMSE (degrees): 8.722293853759766
9.4 Reasoning Regression Model (Text Embedding + Image)
Speed RMSE (mph): 3.264098882675171
Steering RMSE (degrees): 6.014684200286865
Latency (seconds): 0.12727377402571668
9.5 Distilled Regression Model
Speed RMSE (mph): 8.59986400604248
Steering RMSE (degrees): 6.309962749481201
Latency (seconds): 0.049733283448547716

10. Overfitting Analysis
Overfitting was possible given the relatively small dataset size. However, on the dataset used, I did not observe significant overfitting. I used early stopping to ensure good generalization.

11. Iteration and Design Decisions
My initial expectation was that the reasoning model with a regression head on the final token embedding would perform best. While it performed well, incorporating image information improved performance significantly, leading me to iterate on the design.

12. Conclusions
I made substantial progress toward the original goal. I successfully built a model that can predict vehicle speed within approximately 3.25 mph and 6 degrees of the true value on average using chain-of-thought reasoning.
I also explored latency optimization. While the distilled model was roughly three times faster, it performed significantly worse in prediction accuracy. Given the time and resource constraints of this project, I believe the results represent meaningful progress toward end-to-end self-driving reasoning models.

