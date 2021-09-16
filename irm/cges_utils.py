import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch

import random
import numpy as np
import torch
import os

from collections import OrderedDict  
                                                                                                                                                    
# 4-layer                                                                                                                                           
                                                                                                                                                    
glayerwise = [1.,1.0, 1./15, 1./144]                                                                                                                
elayerwise = [1.,0.5, 15., 144.]                                                                                                                    
                                                                                                                                                    
# # 5-layer                                                                                                                                           
                                                                                                                                                    
# glayerwise = [1.,1.0, 1./15, 1./144, 1./144]                                                                                                        
# elayerwise = [1.,0.5, 15., 144., 144.]                                                                                                              
                                                                                                                                                    
def apply_cges(args, model, optimizer):                                                                                                             
    """Applies proximal GD update rule on the weights"""                                                                                            
                                                                                                                                                    
    global glayerwise                                                                                                                               
    global elayerwise                                                                                                                               
                                                                                                                                                    
    learning_rate = optimizer.param_groups[0]['lr']                                                                                                 
                                                                                                                                                    
    glayerwise = glayerwise    
    elayerwise = elayerwise

    S_vars = OrderedDict()
    for key, value in model.state_dict().items():
        if 'weight' in key:
            S_vars[key] = value


    if len(S_vars) > len(glayerwise) or len(S_vars) > len(elayerwise):
        raise Exception("S_vars(length: %d) and layerwise ratios(length: %d / %d) lengths do not match!" %
                         (len(S_vars), len(glayerwise), len(elayerwise)))

    state_dict = model.state_dict()

    for vind, (key, var) in enumerate(S_vars.items()):
        # GS
        group_sum = torch.sum(torch.square(var), 0)
        g_param = learning_rate * args.lamb * (args.mu - vind * args.chvar)
        gl_comp = 1. - g_param * glayerwise[vind] * torch.rsqrt(group_sum)
        gl_plus = (gl_comp > 0).type(torch.float32) * gl_comp
        gl_stack = torch.stack([gl_plus for _ in range(var.shape[0])], 0)
        gl_op = gl_stack * var

        # ES
        e_param = learning_rate * args.lamb * ((1. - args.mu) + vind * args.chvar)
        W_sum = e_param * elayerwise[vind] * torch.sum(torch.abs(gl_op), 0) #Equation 8 of the paper
        W_sum_stack = torch.stack([W_sum for _ in range(gl_op.shape[0])], 0)
        el_comp = torch.abs(gl_op) - W_sum_stack
        el_plus = (el_comp > 0).type(torch.float32) * el_comp

        state_dict[key] = el_plus * torch.sign(gl_op)