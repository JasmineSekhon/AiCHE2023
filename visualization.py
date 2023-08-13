import torch 
import numpy as np 
import pandas as pd
import argparse
import os 
import random 

from models.GDN import GDN

from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc



random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(0)


train_config = {'batch': 32, 
                    'epoch': 30, 
                    'slide_win': 5, 
                    'dim': 64, 
                    'slide_stride': 1, 
                    'comment': 'nawi_data', 
                    'seed': 5, 'out_layer_num': 1, 
                    'out_layer_inter_dim': 128, 
                    'decay': 0.0, 
                    'val_ratio': 0.2, 
                    'topk': 5}
dataset = 'nawi_data'
train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=None)
test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=None)

train, test = train_orig, test_orig

if 'attack' in train.columns:
    train = train.drop(columns=['attack'])

feature_map = get_feature_map(dataset)
fc_struc = get_fc_graph_struc(dataset)


fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

edge_index_sets = []
edge_index_sets.append(fc_edge_index)


model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            )

state_dict = torch.load('./pretrained/nawi_data/best_model.pt')
for param in state_dict:
    if not (state_dict[param].shape == model.state_dict()[param].shape):
        print(param, state_dict[param].shape, model.state_dict()[param].shape)

model.load_state_dict(state_dict)

coeff_weights = model.gnn_layers[0].att_weight_1.cpu().detach().numpy()
edge_index = model.gnn_layers[0].edge_index_1.cpu().detach().numpy()
weight_mat = np.zeros((feature_num, feature_num))

for i in range(len(coeff_weights)):
        edge_i, edge_j = edge_index[:, i]
        edge_i, edge_j = edge_i % feature_num, edge_j % feature_num
        weight_mat[edge_i][edge_j] += coeff_weights[i]

# Next, you could average weight_mat if you use batches or directly use the result if you only use batch=1. 
weight_mat /= train_config['batch']
print(weight_mat.shape)

