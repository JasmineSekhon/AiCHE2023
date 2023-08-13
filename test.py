import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns 

import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *



def test(model, dataloader, save_path=' ', test=False):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    resultLogger = {}
    resultLogger['weightMatrices'] = []
    resultLogger['predictions'] = []
    resultLogger['groundTruth'] = []
    resultLogger['predictedLabels'] = []
    resultLogger['groundTruthLabels'] = []

    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        feature_num = 17

        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            coeff_weights = model.gnn_layers[0].att_weight_1.cpu().detach().numpy()
            edge_index = model.gnn_layers[0].edge_index_1.cpu().detach().numpy()
            weight_mat = np.zeros((feature_num, feature_num))

            for i in range(len(coeff_weights)):
                    edge_i, edge_j = edge_index[:, i]
                    edge_i, edge_j = edge_i % feature_num, edge_j % feature_num
                    weight_mat[edge_i][edge_j] += coeff_weights[i]

            # Next, you could average weight_mat if you use batches or directly use the result if you only use batch=1. 
            weight_mat /= 1
            loss = loss_func(predicted, y)
            resultLogger['weightMatrices'].append(weight_mat)
            resultLogger['predictions'].append(predicted.cpu().numpy())
            resultLogger['groundTruth'].append(y.cpu().numpy())

            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    #print(test_predicted_list)
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    if test:
        print("Saving results to ", f'{save_path}/results_{model.graph_structure}_{model.input_dim}_{model.out_layer_inter_dim}_{model.out_layer_num}_{model.dim}.pth')
        torch.save(resultLogger, f'{save_path}/results_{model.graph_structure}_{model.input_dim}_{model.out_layer_inter_dim}_{model.out_layer_num}_{model.dim}.pth')
    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




