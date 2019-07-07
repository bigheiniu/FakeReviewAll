import sklearn.metrics as metrics


def accuracy_score(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)

def tensor2list(tensor):
    if tensor.requires_grad:
        result = tensor.detach().cpu().numpy().tolist()
    else:
        result = tensor.cpu().numpy().tolist()
    return result