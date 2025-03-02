import pickle
import json
import sys
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)