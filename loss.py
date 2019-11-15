import torch


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, labels):
        pred = self.softmax(pred)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)

        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        ce = (-1*torch.sum(label_one_hot * torch.log(pred), dim=1)).mean()
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        loss = self.alpha * ce + self.beta * rce

        return loss
