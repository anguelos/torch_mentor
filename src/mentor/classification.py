import torchvision
import torch

from .util import current_epoch
from .evaluation import TormetingEvaluator
import tqdm


def render_status(net):
        epoch, val, train = net.status
        return f"E:{epoch:5}, V:{val*100:6.2f}%, T:{train*100:6.2f}%"


def create_classification_model(archname, n_classes, pretrained=True, freeze_layers_before=0, device="cuda"):
        if archname.lower() == "resnet50":
                net=torchvision.models.resnet50(pretrained=pretrained)
                net.fc=torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        elif archname.lower() == "mobilenetv3":
                #len(list(net.parameters())) == 142
                #len(list(net.classifier.parameters())) == 4
                net = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
                net.classifier[-1]=torch.nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        else:
                raise ValueError
        for param in list(net.parameters())[:freeze_layers_before]:
                param.requires_grad = False
        net = net.to(device)
        net.status = (0, 0., 0.) # Epoch, Train Error, Validation Error
        net.args_history = {}
        net.train_history = []
        net.validation_history = {}
        net.best_weights = net.state_dict()
        return net


def iterate_classification_epoch(net, dataloader, evaluator:TormetingEvaluator, loss_fn=None, optimizer=None, device="cuda", verbocity=1):
        is_training = optimizer is not None
        if is_training:
                pass # TODO (anguelos) conditional context manager
                net.train(False)
                caption = f"{render_status} (Training)"
        else:
                net.train(True)
                caption = f"{render_status} (Training)"
        evaluator.reset()
        if verbocity > 0:
                progress_bar = lambda x:tqdm.tqdm(x, desc=caption)
        else:
                progress_bar = lambda x:x
        for inputs, targets in progress_bar(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if is_training:
                        optimizer.zero_grad()
                        predictions = net(inputs)
                        loss = loss_fn(predictions, targets)
                        loss.backward()
                        optimizer.step()
                else:
                        predictions = net(inputs)
                evaluator.update(predictions, targets)
        if is_training:
                net.train_history.append(evaluator.digest())
                net.status = (current_epoch(net), evaluator.single_metric(), net.status[2])
        else:
                net.validation_history[current_epoch(net)]=evaluator.digest()
                net.status = (current_epoch(net), net.status[1], evaluator.single_metric())
