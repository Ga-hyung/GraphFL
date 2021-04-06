import numpy as np

def cora_iid(dataset, num_uers):
    """"
    Sample IID client data from cora dataset
    :param: datasets의 node 갯수
    :param: num_users:
    :return: {client idx : allocated node idx}
    """
    num_node = dataset[0].x
    num_items = int(num_node/num_users)
    dict_users, all_idxs = {}, [i for i in range(num_node)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users