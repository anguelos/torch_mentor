import torchvision
import torch
from .evaluation import TormetingEvaluator


def create_classification_model(archname, n_classes, pretrained=True, freeze_layers_before=0, device="cuda"):
        if archname.lower() == "resnet50":
                net=torchvision.models.resnet50(pretrained=pretrained)
                net.fc=torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        elif archname.lower() == "modilenetv3":
                #len(list(net.parameters())) == 142
                #len(list(net.classifier.parameters())) == 4
                net = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
                net.classifier[-1]=torch.nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        else:
                raise ValueError
        for param in list(net.parameters())[:freeze_layers_before]:
                param.requires_grad = False
        net = net.to(device)
        net.args_history = {}
        net.train_history = []
        net.validation_history = {}
        net.best_weights = net.state_dict()
        return net


def iterate_classification_epoch(net, dataloader, evaluator:TormetingEvaluator, loss_fn=None, optimizer=None, device="cuda"):
        is_training = optimizer is not None
        if is_training:
                pass # TODO (anguelos) conditional context manager
                net.training(False)
        else:
                net.train(True)
        evaluator.reset()
        for inputs, targets in dataloader:
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
                net["train_history"].append(evaluator.digest())
