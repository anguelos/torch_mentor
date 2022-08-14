import torch
from .util import warn, last, current_epoch
from .classification import create_classification_model
from copy import copy
from pathlib import Path
from types import SimpleNamespace

def save(fname,net):
        save_dict = {"weights":net.state_dict(),"args_history":net.args_history, "train_history":net.train_history, "validation_history":net.validation_history, "best_weights":net.best_weights, "status":net.status, "class_names": net.class_names}
        torch.save(save_dict,open(fname,"wb"))


def resume_classification(fname, args=None, device="", arch="", n_classes=0, pretrained=True, freeze_all_before=0):
        if Path(fname).is_file():
                save_dict = torch.load(open(fname,"rb"), map_location="cpu")
                if save_dict["args_history"]:
                        valid_args = copy(last(save_dict["args_history"]))
                else:
                        valid_args = SimpleNamespace()
                valid_args.__dict__.update(args.__dict__)
        else:
                warn(f"could not load {fname}")
                save_dict={}
        if arch == "" and arch in valid_args.__dict__:
                arch = valid_args.arch
        if arch == "":
                arch = "resnet50"
        if device == "" and device in valid_args.__dict__:
                device = valid_args.device
        if device == "":
                device = "cpu"
        if n_classes == 0 and "n_classes" in valid_args.__dict__:
                n_classes = valid_args.n_classes
        if n_classes == 0:
                n_classes = 2
        if pretrained == True and "pretrained" in valid_args.__dict__:
                pretrained = valid_args.pretrained
        if freeze_all_before == 0 and "freeze_all_before" in valid_args.__dict__:
                freeze_all_before = valid_args.freeze_all_before

        net = create_classification_model(arch=arch, n_classes=n_classes, 
                pretrained=pretrained, freeze_layers_before=freeze_all_before, 
                device=device)
        if save_dict:
                net.class_names = save_dict["class_names"]
                net.load_state_dict(save_dict["weights"])
                new_epoch = len(save_dict["train_history"])
                save_dict["args_history"][new_epoch] = args # TODO(anguelos) should we matching history compatibillity?
                net.status = save_dict["status"]
                net.args_history = save_dict["args_history"]
                net.train_history = save_dict["train_history"]
                net.validation_history = save_dict["validation_history"]
                net.best_weights = save_dict["best_weights"]                
        if "class_names" in args.__dict__.keys():  #  TODO (anguelos)  clarify this design pattern
                class_names = tuple(args.class_names.split(","))
                assert (len(net.class_names) == 0) or (class_names == net.class_names)
                net.class_names = class_names
        net = net.to(device)
        if args is not None:
                net.args_history[current_epoch(net)] = args
        return net
