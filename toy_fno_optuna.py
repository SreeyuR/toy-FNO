TEST_MODE = False
num_train_samples = 36_000 # total: 36k
num_test_samples = 4_000  # total: 4k

import torch
from neuralop.models.fno import FNO
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import os
import wandb
import argparse
import optuna
from optuna.trial import TrialState

# don't save locally
os.environ['WANDB_DISABLE_CACHE'] = 'true'

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for toy_FNO')
    parser.add_argument('-n', '--run_name', type=str, default="Default Run", help='Run title')
    parser.add_argument('-trials', '--num_trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('-jobs', '--num_jobs', type=int, default=1, help='Number of parallel jobs')
    args = parser.parse_args()
    return args

def train_model(args, trial=None):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    n_modes = tuple(args.n_modes)
    hidden_channels = args.hidden_channels
    step_size = args.step_size
    gamma = args.gamma
    weight_decay = args.weight_decay
    scheduler_type = args.scheduler
    run_name = args.run_name

    print(f"----------------- Run: {run_name} -----------------")
    # save model
    checkpoint_dir = f'.checkpoints/checkpoints_{run_name}/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # monitor model's progress
    wandb.login()
    # monitor model's progress
    wandb.require("service")
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        reinit=True,
        project="toy-fno",
        group="model-hyperparameter-tuning", # training hyperparameters fixed for now
        name=run_name,
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "FNO",
        "dataset": "toy FNO dataset",
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "scheduler_type": scheduler_type,
        "step_size_lr_scheduler": step_size,
        "gamma_lr_scheduler": gamma,
        "hidden_channels": hidden_channels,
        "n_modes": n_modes,
        }
    )
    
    # Get available GPUs
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = trial.number % len(available_gpus)  # Cycle through available GPUs
    device = torch.device(f"cuda:{available_gpus[gpu_id]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    class L2Loss(object): # L2 norm on functions (normalize by number of grid points)
        # loss returns the sum over all the samples in the current batch
        def __init__(self,):
            super(L2Loss, self).__init__()
        
        def __call__(self, x, y):
            num_examples = x.size()[0]
            diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
            return torch.sum(diff_norms / y_norms)

    #           fourier transform: freq modes                        x & y component of vector field
    model = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=2, out_channels=2) # discretization invariant (uniform grid)
    model.to(device)
    criterion = L2Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Set up scheduler based on user input
    if scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  

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
        epoch_train_loss = running_loss / num_train_samples
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        epoch_test_loss = val_loss / num_test_samples
        test_losses.append(epoch_test_loss)
        
        scheduler.step()  # Update learning rate scheduler
        
        wandb.log({"epoch": epoch, "train_loss": epoch_train_loss, "test_loss": epoch_test_loss, "learning_rate": optimizer.param_groups[0]['lr']}) # log metrics to wandb


        if TEST_MODE:
            print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")
        else: # Save checkpoint
            if (epoch + 1) % 50 == 0:  # Save every 50 epochs
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
    return epoch_test_loss

def objective(trial):
    # fix training params & hypertune model hyperparams, then later hypertune training params
    args = argparse.Namespace(
        run_name=f"trial_model_{trial.number}",
        learning_rate=0.001, 
        batch_size=64, 
        epochs=5, # TODO: Change to 100 later
        n_modes=[trial.suggest_int('n_modes_1', 8, 64), trial.suggest_int('n_modes_2', 8, 64)],
        hidden_channels=trial.suggest_int('hidden_channels', 16, 256),
        step_size=40, 
        gamma=0.5, 
        weight_decay=0.0, 
        scheduler='StepLR' 
    )
   # Alternative way to define the `args` namespace for
   # hyperparameter tuning using Optuna. In this section, the hyperparameters are being tuned using
   # Optuna's suggest methods for different types of parameters:
    # args = argparse.Namespace(
    #     run_name=f"trial_{trial.number}",
    #     learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
    #     batch_size=trial.suggest_categorical('batch_size', [32, 64]),
    #     epochs=100,  # Fixed for now
    #     n_modes=[trial.suggest_int('n_modes_1', 8, 64), trial.suggest_int('n_modes_2', 8, 64)],
    #     hidden_channels=trial.suggest_int('hidden_channels', 16, 256),
    #     step_size=trial.suggest_int('step_size', 10, 100),
    #     gamma=trial.suggest_uniform('gamma', 0.1, 1.0),
    #     weight_decay=trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
    #     scheduler=trial.suggest_categorical('scheduler', ['StepLR', 'ExponentialLR', 'CosineAnnealingLR'])
    # )
    return train_model(args, trial)

def main():
    args = parse_args()
    study = optuna.create_study(
        storage='sqlite:///db.sqlite3', # Specify the storage URL here
        study_name='Toy FNO Optuna recovered',
        load_if_exists=True,
        direction='minimize'
        )
    study.optimize(objective, n_trials=args.num_trials, n_jobs=args.num_jobs, show_progress_bar=True)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
