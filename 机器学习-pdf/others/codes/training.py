import math
import os
import calendar
import time
import torch

def train_nn(nn, train_loader, valid_loader, lossfunction, optimizer):
    training_ID = int(calendar.timegm(time.gmtime()))
    print(f'The ID for this training is {training_ID}.')
    if not os.path.exists('./temp/'):
        os.makedirs('./temp/')

    train_loss = []
    valid_loss = []
    best_valid_loss = math.inf
    patience = 0

    for epoch in range(10**10):
        for x_train, y_train in train_loader:
            prediction_train = nn(x_train)
            L_train = lossfunction(prediction_train, y_train)
            train_loss.append(L_train.item())
            
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()
            
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                L_valid = lossfunction(prediction_valid, y_valid)
                valid_loss.append(L_valid.item())

        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            torch.save(nn, f'./temp/NN_{training_ID}')
            patience = 0
        else:
            patience += 1

        if patience > 5000:
            print('Early stop.')
            break

        if not epoch % 500:
            print(f'| Epoch: {epoch:-8d} | Train loss: {L_train.item():.5f} | Valid loss: {L_valid.item():.5f} |')
    
    resulted_nn = torch.load(f'./temp/NN_{training_ID}')
    os.remove(f'./temp/NN_{training_ID}')
    
    print('Finished.')
    return resulted_nn, train_loss, valid_loss