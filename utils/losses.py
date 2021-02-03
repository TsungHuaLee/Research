# +
import torch.nn as nn
import torch 
import torch.nn.functional as F

class BCEWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return 'bce'
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Variable :math:`(N, 1)`
            y_true: Variable :math:`(N)` where each value is `0 or 1`, torch.FloatTensor
        Returns:
            sigmoid-binary-cross_entropy
        """
        y_pred = y_pred.squeeze()
        y_true = y_true.type(y_pred.dtype)
        assert y_pred.shape == y_true.shape, "input shape {} isn't same as gt shape {}".format(y_pred.shape, y_true.shape)
        return nn.BCEWithLogitsLoss()(y_pred, y_true)
    

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    @property
    def __name__(self):
        return 'CrossEntropyLoss'
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Variable :math:`(N, C)` where `C = number of classes`
            y_true: Variable :math:`(N)` where each value is `0 <= targets[i] <= C-1`, torch.Longtensor
        Returns:
            softmax-cross_entropy
        """
        return nn.CrossEntropyLoss(**self.kwargs)(y_pred, y_true)

class SoftLabelCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    @property
    def __name__(self):
        return 'CrossEntropyLoss'
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Variable :math:`(N, C)` where `C = number of classes`
            y_true: Variable :math:`(N, C)` where each value is torch.Floattensor
        Returns:
            softmax-cross_entropy
        """
        log_likelihood = -1*nn.LogSoftmax(dim=1)(y_pred)
        N, C = y_pred.size()
        loss = torch.sum(torch.mul(log_likelihood, y_true))/N
        return loss
    
class quadratic_kappa(nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 0
        self.den = 0
    @property
    def __name__(self):
        return 'quadratic_kappa'
    def forward(self, y_pred, y_gt):
        y_pred = torch.squeeze(y_pred)
        
        N = len(y_gt)
        weight = torch.zeros((N, N))
        confusion_matrix = torch.zeros((N, N))
        for i in range(N):
            for j in range(N):
                weight[i][j] = ((i-j)**2)/((N-1)**2)
        
        for i, j in zip(y_gt, y_pred):
            confusion_matrix[i][j] += 1
        
        gt_hist = torch.zeros((N))
        for i in y_gt:
            gt_hist[i] += 1
        
        pred_hist = torch.zeros((N))
        for i in y_pred:
            pred_hist[i] += 1
        
        
        E = torch.ger(gt_hist, pred_hist)
        E = E/torch.sum(E)
        
        confusion_matrix = confusion_matrix/torch.sum(confusion_matrix)
        
        self.num = torch.sum(weight * confusion_matrix)
        self.den = torch.sum(weight * E)

        return (1-(self.num/self.den))
