import torch
import torch.nn as nn
import torch.nn.functional as F


class Action_Conditioned_FF(nn.Module):
    """
    Feed-forward neural network that predicts collision probability
    based on 5 distance sensors and the robot's current action.
    """

    def __init__(self, input_sensors=5, num_actions=1):
        super(Action_Conditioned_FF, self).__init__()

        # Total input size = 5 sensor readings + 1 action
        input_dim = input_sensors + num_actions

        # Define the fully connected layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output logit
        )

    def forward(self, sensors, action=None):
        """
        Forward pass.
        Supports two calling styles:
          1. model(sensors, action) → used during training
          2. model(single_tensor)  → used by goal_seeking.py during simulation
        """
        if action is None:
            # Input is already combined [5 sensors + 1 action]
            x = sensors
        else:
            # Concatenate sensors and action along the feature dimension
            x = torch.cat([sensors, action], dim=1)

        return self.net(x)

    @torch.no_grad()
    def evaluate(self, test_loader, loss_function=None, device=None):
        """
        Evaluate the model on a test dataset.
        Returns the average loss and accuracy.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        total_loss, total_correct, total_samples = 0.0, 0, 0

        for sensors, action, labels in test_loader:
            sensors, action, labels = (
                sensors.to(device),
                action.to(device),
                labels.float().to(device).view(-1, 1),
            )

            outputs = self.forward(sensors, action)

            # Compute loss if a function is provided
            if loss_function is not None:
                loss = loss_function(outputs, labels)
                total_loss += loss.item() * sensors.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return avg_loss, accuracy


# --------------------------------------------------
# Quick self-test
# --------------------------------------------------
if __name__ == "__main__":
    model = Action_Conditioned_FF()
    sensors = torch.rand(4, 5)
    actions = torch.rand(4, 1)

    # Training-style call
    out_train = model(sensors, actions)
    print("Training call output shape:", out_train.shape)

    # Inference-style call (used by goal_seeking.py)
    out_infer = model(torch.rand(4, 6))
    print("Inference call output shape:", out_infer.shape)
