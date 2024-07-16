TEST_MODE = False
num_train_samples = 36_000 # total: 36k
num_test_samples = 4_000  # total: 4k

import torch
from neuralop.models.fno import FNO
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import os
import wandb

# monitor model's progress
#wandb.login()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="toy-fno-main",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "FNO",
    "dataset": "???",
    "epochs": 100,
    "batch_size": 256
    }
)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# save model
checkpoint_dir = './checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

# load data
data = torch.load('./Data.pt')
train_input = data["train_in"]
train_output = data["train_sol"]
test_input = data["test_in"]
test_output = data["test_sol"]

print(train_input.shape)  # (N, 2, 64, 64)
print(test_output.shape)  # (N, 2, 64, 64)

train_input = train_input[:num_train_samples]
train_output = train_output[:num_train_samples]
test_input = test_input[:num_test_samples]
test_output = test_output[:num_test_samples]

# define and instantiate: loss and model
class L2Loss(object):
    # loss returns the sum over all the samples in the current batch
    def __init__(self,):
        super(L2Loss, self).__init__()
    
    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        return torch.sum(diff_norms / y_norms)

model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=2, out_channels=2)
model.to(device)
criterion = L2Loss()

# set parameters
epochs = 100
learning_rate = 0.001
batch_size = 256
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # Adjust lr every 50 epochs by multiplying with gamma

train_dataset = TensorDataset(train_input, train_output)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_input, test_output)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train and test model at each epoch
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_samples = len(train_loader) * batch_size # (num_samples / batch size) * batch_size
    epoch_train_loss = running_loss / train_samples
    train_losses.append(epoch_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    epoch_test_loss = val_loss / len(test_loader)
    test_losses.append(epoch_test_loss)
    
    scheduler.step()  # Update learning rate scheduler
    
    wandb.log({"epoch": epoch, "train_loss": epoch_train_loss, "test_loss": epoch_test_loss, "learning_rate": optimizer.param_groups[0]['lr']}) # log metrics to wandb


    if TEST_MODE:
        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")
    else: # Save checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }, checkpoint_path)
            # save model checkpoint to wandb
            artifact = wandb.Artifact('model-checkpoints', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f'Saved checkpoint at epoch {epoch+1}')

# Plot train and test losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss per Epoch')
plt.legend()
if not TEST_MODE:
    plt.savefig('train_test_loss_per_epoch')
else:
    plt.show()

# TODO: modifying hyperparameters: n_modes, width, epochs, learning_rate, batch_size. Grid search?

# Save the losses to a file
if not TEST_MODE:
    losses = {
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    with open('losses.json', 'w') as f:
        json.dump(losses, f)
        
wandb.finish()
