import torch
from .util import warn
from .classification import create_classification_model


def save(fname,net):
        save_dict = {"weights":net.state_dict(),"args_history":net.args_history, "train_history":net.train_history, "validation_history":net.validation_history, "best_weights":net.best_weights, "status":net.status}
        torch.save(save_dict,open(fname,"wb"))


def resume_classification(args, fname):
        net = create_classification_model(archname=args.arch, n_classes=args.n_classes, pretrained=args.pretrained, freeze_layers_before=args.freeze_all_before, device=args.device)
        try:
                save_dict = torch.load(open(fname,"rb"), map_location=torch.device(args.device))
                net.load_state_dict(save_dict["weights"])
                new_epoch = len(save_dict["train_history"])
                save_dict["args_history"][new_epoch] = args
                net.status = save_dict["status"]
                net.args_history = save_dict["args_history"]
                net.train_history = save_dict["train_history"]
                net.validation_history = save_dict["validation_history"]
                net.best_weights = save_dict["best_weights"]

        except FileNotFoundError:
                warn(f"could not load {fname}")
        return net
