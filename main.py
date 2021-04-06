from model import GCN
from utils import get_dataset
from update import LocalUpdate, test_inference
import os
import copy
import time
import pickle
import torch
import torch.nn.functional as F
import numpy as np


if __name__=='__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_datset, user_groups = get_dataset(args)

    # set the model to train and send it to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = GCN().to(device)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01, weight_decay=5e-4)

    global_model.train()
    global_weights = global_model.state_dict()

    #Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [],[]
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in range(200):
        local_weights, local_losses = [], []
        print("f\n |Global Training Round: {epoch+1} |\n")
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs=user_groups[idx], logger = logger)
            w, loss = local_model.update_weights(model = copy.deepcopy(global_model), global_round = epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        #update global weights
        global_weights = average_weights(local_weights) #########################AVG weight하는 부분 check

        #update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses)/len(local_losses)
        train_loss.append(loss_avg)

        #Calculate avg training accuracy over all users at every epoch

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args = args, dataset = train_dataset, idxs = user_groups[idx], logger = logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        #print global training loss after every i round
        if (epoch+1) % print_every ==0:
            print(f"\n Avg Training Stats after {epoch+1} global rounds")
            print(f'Training loss: {np.mean(np.array(train_loss))}')
            print("Train Accuracy: {:.2f}% \n".format(100*train_accuracy[-1]))

    #Test Inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f"\n Results after {args.epochs} global rounds of training")
    print("| --- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[--1]))
    print("| -- Test Accuracy: {:.2f}%".format(100*test_acc))




