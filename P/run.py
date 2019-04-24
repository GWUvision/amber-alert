from _code.G_train import RunTrain
from _code.G_test import RunTest
from _code.G_eval import RunAcc
from _code.G_similarity import RunSimilarity
from glob import glob
import os, torch

###### CAR resnet18 256 by 256 ###### 
Data='CAR'## calulate dataset RGB MEAN and STD first and modify _code.colorlib file
src='/pless_nfs/home/krood20/devinkopp/P/images/'##
data_dict = {p:{os.path.basename(d):sorted(glob(d+'/*.png')) for d in glob(src+p+'/*')} for p in ['tra', 'val']}

for threshold in [0.5]:
    for d,l in [(64,1)]:
        print('dimension = {}'.format(d))
        dst = '_result/'
        #print(dst)
        #RunTrain(Data, dst, data_dict, emb_dim=d*l, th=threshold, pb=[d,l])
        #RunTest(Data,  dst, data_dict['tra'], 255, 'tra', pb=[d,l])
        RunTest(Data,  dst, data_dict['val'], 255, 'val', pb=[d,l])
        #RunAcc(dst)
	#RunSimilarity(dst)
