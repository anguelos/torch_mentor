import sklearn
import torch


class TormetingEvaluator():
        def reset(self):
                pass

        def update(self, *batch_args):
                raise NotImplementedError

        def digest(self)->dict:
                return {"Value":0.}
        
        def __str__(self):
                raise NotImplementedError

        def single_metric(self):
                raise NotImplementedError


class TwoClassEvaluator():
        def __init__(self, loss_fn=None, roc_step=.01):
                self.loss_fn = loss_fn
                self.reset()

        def reset(self):
                self.y_true = []
                self.y_score = []

        def update(self, predictions, targets):
                self.y_score.append(predictions.detach())
                self.y_true.append(targets.detach())
        
        def digest(self):
                result = {}
                y_score = torch.cat(self.y_score, dim=0)
                y_true = torch.cat(self.y_true, dim=0)
                if self.loss_fn is not None:
                        with torch.no_grad():
                                losses = self.loss_fn(y_score, y_true).sum()
                        result["loss"] = losses.cpu().numpy()
                y_score, y_train = y_score.cpu().numpy(), y_train.cpu().numpy() 
                roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
                accuracy = ((y_score>.5) == y_true).mean()
                result.update({"ROC AUC": roc_auc, "Accuracy": accuracy})
                return result
        
        def __str__(self):
                outputs = self.digest()
                return repr(outputs)

        def single_metric(self):
                self.digest()['Accuracy']
                
                
