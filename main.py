import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1. Load and preprocess the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the labels, ensuring the tensor is of type long
y = torch.tensor(y, dtype=torch.long)  # Convert to long type before one-hot encoding
y_one_hot = torch.nn.functional.one_hot(y, num_classes=3).float()  # One-hot encode and convert to float

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = y_train
y_test_tensor = y_test

# Create DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 2. Define the PyTorch Lightning model
class IrisModel(pl.LightningModule):
    def __init__(self):
        super(IrisModel, self).__init__()
        # Define layers: input -> hidden -> output
        self.layer1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 3)  # 3 classes
        
        # Initialize lists to store validation metrics
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        return self.output(x)

    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True)
        
        # Monitor validation loss for scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # We monitor validation loss for ReduceLROnPlateau
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, torch.argmax(y, dim=1))  # Using CrossEntropyLoss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, torch.argmax(y, dim=1))
        acc = (y_hat.argmax(dim=1) == torch.argmax(y, dim=1)).float().mean()
        
        # Log the metrics (loss and accuracy) to trainer.callback_metrics
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        # You can access val_loss and val_acc here directly via trainer.callback_metrics
        val_loss = self.trainer.callback_metrics.get('val_loss').cpu().item()
        val_acc = self.trainer.callback_metrics.get('val_acc').cpu().item()
        
        # Save to instance variables for later plotting
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

    # Define test_step() method for evaluation
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, torch.argmax(y, dim=1))
        acc = (y_hat.argmax(dim=1) == torch.argmax(y, dim=1)).float().mean()
        
        # Log test loss and accuracy
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # Replace test_epoch_end with on_test_epoch_end (for PyTorch Lightning v2.0.0+)
    def on_test_epoch_end(self):
        # Aggregate test results
        test_loss = self.trainer.callback_metrics.get('test_loss').cpu().item()
        test_acc = self.trainer.callback_metrics.get('test_acc').cpu().item()

        # Optionally log aggregated results
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)

# 3. Initialize the model
model = IrisModel()

# 4. Initialize the PyTorch Lightning trainer with early stopping
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping])

# 5. Train the model
trainer.fit(model, train_loader, test_loader)

# 6. Evaluate the model on the test set
trainer.test(model, test_loader)

# 7. Get the training and validation history (loss and accuracy)
# Note: This is now saved within the model itself via on_validation_epoch_end

# 8. Plot the loss and accuracy curves
plt.figure(figsize=(12, 5))

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(model.val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(model.val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
