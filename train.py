#%%
from di_dataset import DeepInsightDataset
from model import DeepInsightVitModel
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
import numpy as np
import torch

# TODO - define train loop
def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    lr=0.0001,
    epochs=20,
    optimiser=torch.optim.Adam,
    ):
    # init tensorboard:
    writer = SummaryWriter()
    scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=[5,10], gamma=0.1,verbose=True)
    optimiser = optimiser(model.parameters(), lr=lr, weight_decay=0.001)
    batch_idx = 0
    epoch_idx= 0

    for epoch in range(epochs):  # for each epoch
        # 
        
        print('Epoch:', epoch_idx,'LR:', scheduler.get_lr())
        # weights_filename=model.weights_folder_name + '_latest_weights.pt'
        epoch_idx +=1
        # torch.save(model.state_dict(), weights_filename)
        for batch in train_loader: 
            features, labels = batch
            # print(size(images))
            
            # forward pass
            output1, output2 = model(features)  
            # compare the predictions of each regression head to their respective labels
            loss = model.combined_loss(output1, output2, labels)
            loss.backward()  # calculate the gradient of the loss with respect to each model parameter
            optimiser.step()  # use the optimiser to update the model parameters using those gradients
            print("Epoch:", epoch, "Batch:", batch_idx,
                  "Loss:", loss.item())  # log the loss
            optimiser.zero_grad()  # zero grad
            writer.add_scalar("Loss/Train", loss.item(), batch_idx)
            batch_idx += 1
            
        print('Evaluating on valiudation set')
        # evaluate the validation set performance
        val_MSE = evaluate(model, val_loader)
        writer.add_scalar("Val_MSE", val_MSE, batch_idx)
        scheduler.step()
    # evaluate the final test set performance
    
    print('Evaluating on test set')
    test_loss = evaluate(model, test_loader)
    # writer.add_scalar("Loss/Test", test_loss, batch_idx)
    model.test_loss = test_loss
    
    return model   # return trained model
    

def evaluate(model, dataloader):
    losses = []
    correct = 0
    n_examples = 0
    for batch in dataloader:
        features, labels = batch
        output1, output2 = model(features)  
        loss = model.combined_loss(output1, output2, labels)
        losses.append(loss.detach())
        # TODO - work out correct accuracy metric here:
        

    avg_loss = np.mean(losses)
    print("Loss:", avg_loss)
    return avg_loss

if __name__  == "__main__":
    dataset = DeepInsightDataset()
    train_set_len = round(0.8*len(dataset))
    val_set_len = round(0.1*len(dataset))
    test_set_len = len(dataset) - val_set_len - train_set_len
    split_lengths = [train_set_len, val_set_len, test_set_len]
    # split the data to get validation and test sets
    train_set, val_set, test_set = random_split(dataset, split_lengths)

    batch_size = 16
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = DeepInsightVitModel

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=20,
        lr=0.0001,
        optimiser=torch.optim.AdamW
        
    )



# %%
