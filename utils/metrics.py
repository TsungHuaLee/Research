# +
import torch.nn as nn
import torch 
import torch.nn.functional as F

import sklearn
class AUC(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return 'auc'
    def forward(self, y_pred, y_gt):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_gt, y_pred, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        return roc_auc

class Fscore(nn.Module):
    def __init__(self, beta=1, epsilon=1e-7, threshold=0.5):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.threshold = threshold
    @property
    def __name__(self):
        return 'fscore'    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Variable :math:`(N, C)` where `C = number of classes`
            y_true: Variable :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        Returns:
            fscore
        """
        assert y_true.ndim == 1
        assert y_pred.ndim == 2
        assert y_pred.shape[1] == 2 or y_pred.shape[1] == 1 # final layer two neuron or one neuron
        
        """ ignore unlabled data"""
        non_negative_index = y_true != -1
        y_true = y_true[non_negative_index]
        y_pred = y_pred[non_negative_index]
        
        if y_pred.shape[1] == 2:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > self.threshold).type(y_pred.dtype)
            y_pred = y_pred.squeeze()
        
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        
        score = ((1 + self.beta ** 2) * tp + self.epsilon) \
            / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.epsilon)
        return score

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return 'accuracy'
    def forward(self, y_pred, y_gt):
        y_pred = torch.softmax(y_pred, dim = 1)
        y_pred = torch.argmax(y_pred, axis = 1)
        tp = torch.sum(y_gt == y_pred, dtype=y_pred.dtype)
        tp = tp.type(torch.DoubleTensor)
        score = tp/y_gt.view(-1).shape[0]
        return score
