import glob
from collections import defaultdict 


def get_feature_map(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list

def get_apriori_fc_graph_struc(dataset):
    if 'modified' in dataset:
        feature_file = open(f'./data/{dataset}/list.txt', 'r')
        feature_list = []
        for ft in feature_file:
            feature_list.append(ft.strip())

        struc_map = {
            'PT2': [],
            'PT5': ['FT2', 'FT1'],
            'FT0': ['PT2'],
            'FT1': ['FT2', 'FT0'],
            'FT2': ['FT0'],
            'NP': ['PF', 'T', 'mean(PT3, PT4)', 'PT5'],
            'SP': ['CT2', 'CT1'],
            'CT1': [],
            'CT2': ['CT1', 'T', 'PF', 'FT1', 'FT2'],
            'T': [],
            'PF': ['CT1', 'T', 'PT5', 'FT1', 'mean(PT3, PT4)'],
            'mean(PT3, PT4)': ['PT2', 'FT2', 'FT1']
        }

    else:

        feature_file = open(f'./data/{dataset}/list.txt', 'r')
        feature_list = []
        for ft in feature_file:
            feature_list.append(ft.strip())

        struc_map = {
            'PT2': [],
            'PT3': ['PT2', 'FT2', 'FT1'],
            'PT4': ['FT2', 'PT3'],
            'PT5': ['FT2', 'FT1'],
            'FT0': ['PT2'],
            'FT1': ['FT2', 'FT0'],
            'FT2': ['FT0'],
            'FT3': ['CT1', 'T', 'PT4', 'PT5', 'FT1', 'PT3'],
            'NT1': ['PF', 'T', 'PT3', 'PT4', 'PT5'],
            'NP': ['NT1'],
            'SP': ['CT2', 'CT1'],
            'CT1': [],
            'CT2': ['CT1', 'T', 'PF', 'FT1', 'FT2'],
            'T': [],
            'PF': ['CT1', 'T', 'PT4', 'PT5', 'FT1', 'PT3'],
        }
    
    new_struc_map = {}
    for k in struc_map:
        if k in feature_list:
            new_struc_map[k] = struc_map[k]
    return new_struc_map
    

# graph is 'fully-connect'
def get_fc_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    
    return struc_map

def get_prior_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/features.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    
    return struc_map


if __name__ == '__main__':
    get_graph_struc()
 
