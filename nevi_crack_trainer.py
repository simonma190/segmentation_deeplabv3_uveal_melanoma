import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log_1212_test.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    best_auc_test = 0
    best_auc_train = 0
    best_auc_train_ave = 0
    best_auc_test_ave = 0

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()


                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
#                    loss = criterion(outputs, masks)


                    y_pred_0 = outputs['out'][0].argmax(0).data.cpu().numpy().ravel()
                    y_true_0 = masks.data[0].argmax(0).cpu().numpy().ravel()
                    y_pred_1 = outputs['out'][1].argmax(0).data.cpu().numpy().ravel()
                    y_true_1 = masks.data[1].argmax(0).cpu().numpy().ravel()
                    y_pred = np.concatenate((y_pred_0, y_pred_1), axis=None)
                    y_true = np.concatenate((y_true_0, y_true_1), axis=None)

                    #fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                    #optimal_idx = np.argmax(tpr - fpr)
                    #optimal_threshold = thresholds[optimal_idx]
                    #print("Threshold value is:", optimal_threshold)
                    #print("Sensitivity is:", tpr[optimal_idx])
                    #print("Specificity is:", 1-fpr[optimal_idx])
                    for name, metric in metrics.items():
                        if name == 'f1_score_a':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true, y_pred, average=None)[0])
                        elif name == 'f1_score_b':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true, y_pred, average=None)[1])
                        if name == 'f1_score_c':
                            batchsummary[f'{phase}_{name}'].append(
                                #metric(y_true, y_pred, average=None)[2])
                                0.1)
                        else:
                            #b += metric(y_true.astype('uint8'), y_pred)
                            batchsummary[f'{phase}_{name}'].append(
                               0.1)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
          
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
            #print(a)
            #print(b)
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        best_auc_train_temp = batchsummary['Train_auroc']
        best_auc_test_temp = batchsummary['Test_auroc']
        if best_auc_train_ave < best_auc_train_temp:
            best_auc_train_ave = best_auc_train_temp
        if best_auc_test_ave < best_auc_test_temp:
            best_auc_test_ave = best_auc_test_temp        

        print(batchsummary)
        with open(os.path.join(bpath, 'log_1212_test.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))
    print('best_auc_train_ave: {:4f}'.format(best_auc_train_ave))  
    print('best_auc_test_ave: {:4f}'.format(best_auc_test_ave))                          
#    print('Highest Train auc_ave: {:4f}'.format(best_auc_train))
#    print('Highest Test auc_ave: {:4f}'.format(best_auc_test))    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
