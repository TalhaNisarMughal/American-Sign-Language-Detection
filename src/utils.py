import torch
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import os 
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def save_model(model, target_dir, model_name):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)

def train_step(model, dataloader, loss_fn, optimizer, device):
    # Put model on train mode
    model.train()

    # Setup train loss and accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataloader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)   # Send data to target device

        y_pred = model(X)                   # Forward pass
        loss = loss_fn(y_pred, y)           # Calculate loss
        train_loss += loss.item()           # Accumulate loss
        optimizer.zero_grad()               # Optimizer zero grad
        loss.backward()                     # Loss backward
        optimizer.step()                    # Optimizer step

        # Calculate and ammulate accuracy metrics across all batches 
        y_pred_class = torch.argmax(torch.softmax(y_pred,
                                                dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    # Adjust metrics to get avg. loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    # Put model on evaluation mode
    model.eval()

    # Setup test loss and accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through dataloader data batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)       # Send data to target device

            test_pred_logits = model(X)             # Forward pass
            loss = loss_fn(test_pred_logits, y)     # Calculate loss
            test_loss += loss.item()                # Accumulate loss

            # Calculate and ammulate accuracy metrics across all batches 
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)
    
    # Adjust metrics to get avg. loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model, train_dataloader, test_dataloader, 
          loss_fn, optimizer, epochs, device):
    # Create empty results dictionary 
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Loop through train_steps and test_steps for no of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer,
                                           device)
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn,
                                        device)

        # Print out metrics
        print(f"\nEpoch: {epoch+1} | Train loss: {train_loss:.4f} - Train acc: {(train_acc*100):.2f}% -- Test_loss: {test_loss:.4f} -- Test_acc: {(test_acc*100):.2f}%")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

def create_dataloaders(source_dir, transform, batch_size, num_workers):
    # Use ImageFolder to create datasets
    source_data = datasets.ImageFolder(root = source_dir,
                                       transform = transform) # Transforms input data into tensors

    # Get the class names
    class_names = source_data.classes

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(source_data))
    test_size = len(source_data) - train_size
    train_data, test_data = random_split(source_data, [train_size, test_size])

    # Turn images into dataloaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=num_workers, 
                                 pin_memory=True)

    return train_dataloader, test_dataloader, class_names