import sklearn.metrics
import torch


class TormetingEvaluator():
        def reset(self):
                pass

        def update(self, *batch_args):
                raise NotImplementedError

        def digest(self)->dict:
                raise NotImplementedError
                return {"Value":0.}
        
        def __str__(self):
                outputs = self.digest()
                "\n".join([f"{k}:{v}" for k, v in sorted(outputs.items())])
                return repr(outputs)

        def single_metric(self):
                raise NotImplementedError


class TwoClassEvaluator():
        def __init__(self, loss_fn=None, roc_step=.01):
                self.loss_fn = loss_fn
                self.reset()

        def reset(self):
                self.y_true = []
                self.y_pred = []

        def update(self, predictions, targets):
                
                self.y_pred.append(predictions.detach())
                self.y_true.append(targets.detach())
        
        def digest(self):
                result = {}
                y_pred = torch.cat(self.y_pred, dim=0)
                y_true = torch.cat(self.y_true, dim=0)
                if self.loss_fn is not None:
                        with torch.no_grad():
                                losses = self.loss_fn(y_pred, y_true).sum()
                        result["loss"] = losses.cpu().numpy()
                y_pred = torch.nn.functional.softmax(y_pred,dim=1)[:, 0]
                y_true = y_true.float()
                y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy() 
                roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
                tp = ((y_pred==y_true) & (y_pred>.5)).sum()
                fp = ((y_pred!=y_true) & (y_pred>.5)).sum()
                tn = ((y_pred==y_true) & (y_pred<.5)).sum()
                recall = tp /(tp+tn)
                precision = tp /(tp+fp)
                f1 = (2*recall*precision)/(recall+precision)

                #f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_score)
                accuracy = ((y_pred>.5) == y_true>.5).mean()
                result.update({"ROC AUC": roc_auc, "Accuracy": accuracy, "F1":f1, "recall":recall, "precision":precision})
                return result
        
        def __str__(self):
                outputs = self.digest()
                "\n".join([f"{k}:{v}" for k, v in sorted(outputs.items())])
                return repr(outputs)


        def __repr__(self):
                outputs = self.digest()
                return repr(outputs)


        def single_metric(self):
                return self.digest()['Accuracy']
                
                
