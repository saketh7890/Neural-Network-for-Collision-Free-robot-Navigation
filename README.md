

This project develops and trains a neural network model that enables a robot to navigate within a simulated maze environment without colliding with walls. The goal is to integrate perception, learning, and control into an intelligent navigation system that can generalize safe motion behavior across different layouts.

Project Overview

A simulation environment was built to represent the maze and robot movement dynamics. The robot receives sensory readings (distances from walls, obstacle proximity, and movement direction) and outputs an action decision. Each action is labeled as safe or collision-prone, forming the training dataset for the neural network.

A separate goal-seeking simulation demonstrates the robot moving toward a target while applying the trained model’s predictions to ensure wall-free motion.

Model Training

Dataset:
Each data sample contains 5 sensor inputs, 1 action input, and 1 collision label (0 = safe, 1 = collision).

Architecture:
A feed-forward neural network (Action_Conditioned_FF) trained using PyTorch to classify whether a given action is safe.

Training Setup:
Weighted sampling handles class imbalance; training uses AdamW optimization with OneCycleLR scheduling and early stopping.

Metrics:
Validation loss and F1-score monitor performance to ensure generalization and collision-free behavior.

Integration with Simulation

After training, the model is deployed inside a custom simulation environment where the robot autonomously moves toward its goal. The model predicts safe actions in real time, preventing collisions with maze walls even in unseen configurations.

Technical Stack

Languages/Libraries: Python, PyTorch, NumPy, OpenCV, Matplotlib

Learning Technique: Supervised learning / binary classification

Hardware (optional extension): Dobot Magician robotic arm for real-world validation

Outcome

The trained model demonstrates intelligent, collision-aware behavior, allowing the robot to navigate safely and efficiently through maze-like environments.



🔄 Workflow Summary

Prepare Data:
Collect or verify the dataset in saved/training_data.csv (11,000 samples).

Train Model:
Run train_model.py to train and save the collision-avoidance model.

Run Simulation:
Run goal_seeking.py to test the model in a maze environment.
