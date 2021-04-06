import copy
import torch
from torch_geometric import Planetoid
from sampling import cora_iid

def load_cora():
    dataset = Planetoid(root='/tmp/Cora', name = 'Cora', split = 'random', num_train_per_class = 200, num_val = 0, num_test = 1000 )
    cora = dataset[0]
    return cora

def get_dataset(args):
    """
    Returns train and test datasets and user group which is a dict where the keys are the user index and the values are
    corresponding data for each of those users
    """
    dataset = load_cora()

    train_dataset = dataset.x[dataset.train_mask]
    test_dataset = dataset.x[dataset.test_mask]

    user_groups = cora_iid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key]+=w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
