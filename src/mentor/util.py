import torch
import sys


def is_tormenting(net: torch.nn.Module):
        """Validates if a torch.nn.Module is a valid tormenting module"""
        return all([field in net.__dict__ for field in ("args_history", "train_history", "validation_history", "best_weights")])


def last(some_dict:dict):
        return some_dict[sorted(some_dict.keys())[-1]]


def current_epoch(net:torch.nn.Module):
        return len(net.train_history)


def warn(*args):
        for item in args:
                sys.stderr.write(str(item))
        sys.stderr.write("\n")
        sys.stderr.flush()

