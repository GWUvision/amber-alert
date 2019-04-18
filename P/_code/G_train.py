from .color_lib import RGBmean,RGBstdv
from .Train import learn

import os, torch, random

def RunTrain(Data, dst, data_dict, model_name='r50', bt=128, emb_dim=64, sigma=0.1, th=0.5, pb=[0,0], core=[0]):
    if not os.path.exists(dst): os.makedirs(dst)
    x = learn(core, dst, Data, data_dict, model_name=model_name, batch_size=bt)
    # run with partition-based N-pair loss
    x.runNpairsPB(emb_dim, th, pb, sigma=sigma, num_epochs=10)
    # run with standard N-pair loss
    #x.runNpairs(emb_dim, sigma=sigma, num_epochs=10)
    
