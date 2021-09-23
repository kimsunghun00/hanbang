import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
import copy
from sklearn.metrics import confusion_matrix, classification_report


class Trainer:
    def __init__(self, net, train_loader, test_loader, criterion, optimizer, epochs=100, lr=0.001, l2_norm=None, device=None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), lr=lr)
        self.epochs = epochs
        self.lr = lr
        self.l2_norm = l2_norm

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_acc = []

        self.best_model_wts = copy.deepcopy(self.net.state_dict())
        self.best_acc = 0.

    def fit(self):
        for epoch in range(self.epochs):
            running_loss = 0.
            self.net.train()
            n = 0
            n_acc = 0

            for i, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.to(self.device, dtype=torch.float)
                y_batch = y_batch.to(self.device, dtype=torch.int64)
                y_pred_batch = self.net(X_batch)

                # regularization
                if self.l2_norm is not None:
                    lambda2 = self.l2_norm
                    fc_params = torch.cat([x.view(-1) for x in self.net.out.parameters()])
                    l2_regularization = lambda2 * torch.norm(fc_params, p=2)
                else:
                    l2_regularization = 0.

                loss = self.criterion(y_pred_batch, y_batch) + l2_regularization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n += len(X_batch)
                _, y_pred_batch = y_pred_batch.max(1)
                n_acc += (y_batch == y_pred_batch).float().sum().item()
            self.train_losses.append(running_loss / i)
            self.train_acc.append(n_acc / n)

            # 검증 데이터의 예측 정확도
            self.val_acc.append(self.eval_net(self.test_loader, self.device)[0])
            self.val_losses.append(self.eval_net(self.test_loader, self.device)[1])

            # epoch 결과 표시
            print(
                'epoch: {}/{}, train_loss: {:.4f}, train_acc: {:.2f}%, test_acc: {:.2f}%'.format(epoch + 1, self.epochs,
                                                                                                 self.train_losses[-1],
                                                                                                 self.train_acc[
                                                                                                     -1] * 100,
                                                                                                 self.val_acc[
                                                                                                     -1] * 100))
        print('best acc : {:.2f}%'.format(self.best_acc * 100))

    def eval_net(self, data_loader, device):
        self.net.eval()
        ys = []
        y_preds = []
        running_loss = 0.
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.int64)

            with torch.no_grad():
                loss = self.criterion(self.net(X_batch), y_batch)
                _, y_pred_batch = self.net(X_batch).max(1)
            ys.append(y_batch)
            y_preds.append(y_pred_batch)
            running_loss += loss.item()

        ys = torch.cat(ys)
        y_preds = torch.cat(y_preds)
        val_loss = running_loss / i

        acc = (ys == y_preds).float().sum() / len(ys)

        if acc.item() > self.best_acc:
            self.best_acc = acc
            self.best_model_wts = copy.deepcopy(self.net.state_dict())

        return acc.item(), val_loss

    def evaluation(self, data_loader, device):
        model = self.get_best_model()
        ys = []
        y_preds = []
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.int64)

            with torch.no_grad():
                _, y_pred_batch = model(X_batch).max(1)
            ys.append(y_batch)
            y_preds.append(y_pred_batch)

        ys = torch.cat(ys)
        y_preds = torch.cat(y_preds)

        acc = (ys == y_preds).float().sum() / len(ys)

        print("Confusion Matrix")
        print(confusion_matrix(ys.to('cpu'), y_preds.to('cpu')))
        print("Classification Report")
        print(classification_report(ys.to('cpu'), y_preds.to('cpu'), digits=4))

        return acc.item()

    def history(self):
        return {'train_acc' : self.train_acc, 'val_acc' : self.val_acc,
                'train_loss' : self.train_losses, 'val_loss' : self.val_losses}

    def get_best_model(self):
        self.net.load_state_dict(self.best_model_wts)
        return self.net