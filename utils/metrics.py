# +
import torch.nn as nn
import torch 

class CrossEntropy(nn.Module):
    def __init__(self, weight = None):
        super().__init__()
        self.weight = weight
    @property
    def __name__(self):
        return 'cross entropy'
    def forward(self, y_pred, y_gt):
        if self.weight is None:
            return nn.CrossEntropyLoss()(y_pred, y_gt)
        else:
            self.weight = torch.tensor(self.weight).cuda()
            return nn.CrossEntropyLoss(weight=self.weight)(y_pred, y_gt)

class BiFscore(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    @property
    def __name__(self):
        return 'fscore'    
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
    
class Specificity(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7
    @property
    def __name__(self):
        return 'specificity'
    def forward(self, pred, gt):
        pred = torch.softmax(pred, dim = 1)
        pred = torch.argmax(pred, dim = 1)
        
        tp = (gt * pred).sum().to(torch.float32)
        tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
        fp = ((1 - gt) * pred).sum().to(torch.float32)
        fn = (gt * (1 - pred)).sum().to(torch.float32)

        return (tn + self.eps)/(fp + tn + self.eps)
    
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
    
class L1Loss(nn.Module):
    def __init__(self, max_value = 1):
        super().__init__()
        self.max_value = max_value
    @property
    def __name__(self):
        return 'l1loss'
    def forward(self, y_pred, y_gt):
        y_pred =nn.ReLU()(y_pred)
        y_pred = torch.squeeze(y_pred)
        return nn.L1Loss()(y_pred, y_gt)
    
class MSELoss(nn.Module):
    def __init__(self, max_value = 1):
        super().__init__()
        self.max_value = max_value
    @property
    def __name__(self):
        return 'mseloss'
    def forward(self, y_pred, y_gt):
        y_pred = nn.ReLU()(y_pred)
        y_pred = torch.squeeze(y_pred)
        return nn.MSELoss()(y_pred, y_gt)
    
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