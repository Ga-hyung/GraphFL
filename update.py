import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """"
    An abstract Dataset class wrapped around Pytorch Dataset class
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        node, label = self.dataset.x, self.dataset.y

class LocalUpdate(object):
    def __init__(self, args, dataset, idx, logger):
        self.args = args
        self.logger = logger
        self.trainloader,  self.testloader = self.train_val_test(dataset, list(idx))
        self.device = 'cuda' if args.gpu else 'cpu'

        #Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        "Return train, validation and test dataloaders for a given dataset and user indexes."

        #split indexes for train, validation and test (90, 10)

        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size = self.args.local_bs, shuffle = True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size = int(len(idxs_test)/10), shuffle=False)

        return trainloader, testloader

    def update_weights(selfself, model, global_round):
        model.train()
        epoch_loss = []

        optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (node, label) in enumerate(self.trainloader):
                node, label = node.to(self.device), label.to(self.device)

                model.zero_grad()
                log_probs = model(node)
                loss = self.criterion(log_probs, label)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx %10 ==0):
                    print("|Global Round: {} | Local Epoch: {} | [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                        global_round, iter, batch_idx*len(node), len(self.trainloader.dataset),
                        100*batch_idx/len(self.trainloader), loss.item()))
                self.logger.add_scalear('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (node, label) in enumerate(self.testloader):
            node, label = node.to(self.device), label.to(self.device)

            # Inference
            outputs = model(node)
            batch_loss = self.criterion(outputs, label)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, label)).item()
            total += len(label)

            accuracy = correct / total
            return accuracy, loss

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)

    for batch_idx, (node, label) in enumerate(testloader):
        node, label = node.to(device), label.to(device)

        # Inference
        outputs = model(node)
        batch_loss = criterion(outputs, label)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, label)).item()
        total += len(label)
        accuracy = correct / total
    return accuracy, loss
