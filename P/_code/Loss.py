import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

from .Utils import norml2, norml2Galaxy

eps = 0.00001

def distMC(Mat_A, Mat_B, norm=1, cpu=False, sq=True):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    DC = Mat_A.mm(torch.t(Mat_B))
    if cpu:
        if sq:
            DC[torch.eye(N_A).byte()] = -norm
    else:
        if sq:
            DC[torch.eye(N_A).byte().cuda()] = -norm
            
    return DC

def Mat(Lvec):
    N = Lvec.size(0)
    M = torch.zeros(N,N).cuda()
    I = torch.eye(N).cuda()
    
    for l in list(set(Lvec.data.tolist())):
        v = (Lvec==l).view(-1,1).data.float()
        M += (v.mm(torch.t(v)))
        
    return (M-I).byte(), (1-M).byte()#same diff

class NpairsLossPB(Module):
    def __init__(self,s,th,pb=[0,0]):
        super(NpairsLossPB, self).__init__()
        self.d = pb[0]
        self.l = pb[1]
        self.sigma = s
        self.th = 0
        
    def forward(self, fvec, Lvec):
        N = Lvec.size(0)
        idx = [i for i in range(N) if i%2==0 and Lvec[i]!=-1]
        M = torch.ones(N,N).byte().cuda()
        # M[idx,:]=1
        
        
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        
        # New Distance Metric
        D_sum = 0
        M_sum = 0
        for i in range(self.l):
            sta = i*self.d
            end = (i+1)*self.d
            
            fvec_tmp = norml2(fvec[:,sta:end])

            Dist = distMC(fvec_tmp,fvec_tmp)
                
            # relative margin
            Dist_temp = Dist+0.00001
            
            Dist_temp[Diff]=-1
            A,B = Dist_temp.max(1)
            M_temp = torch.ones(N,N).byte().cuda()
            M_temp[torch.LongTensor([i for i in range(N)]),B]=0
            
            Dist_max =A.repeat(Dist.size(1),1).t()#-self.mg
            
            Mask = M_temp*Same
            if self.l>=1: Dist[Mask]=0
            
            # # zeroing operation
            # Mask = (Dist<self.th)
            # if self.l>1: Dist[Mask]=0
            
            D_sum += Dist
            M_sum += Mask.float()
            
        D_sum /= self.l
        M_sum /= self.l
        
        # loss
        D = -F.log_softmax(D_sum/(self.sigma),dim=1)
        loss = D[Same*(1-M_temp)].sum()
        print('loss:{:.4f}\n'.format(loss.item()/N))
        
        return loss, M_sum[Same].mean().cpu(), M_sum[Diff].mean().cpu()

class NpairsLoss(Module):
    def __init__(self,s,th=0.5):
        super(NpairsLoss, self).__init__()
        self.sigma = s
        self.th = th

    def forward(self, fvec, Lvec):
        N = Lvec.size(0)
        idx = [i for i in range(N) if i%2==0]

        M = torch.ByteTensor(N,N).cuda()
        M[idx,:]=1
        
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        
        # normalization
        fvec = norml2(fvec)
        std = (1-fvec.std(dim=0))
        
        # distance matrix
        D_sum = distMC(fvec,fvec)
        M_sum = (D_sum<self.th).float()
        
        # loss
        loss = (-F.log_softmax(D_sum/self.sigma,dim=1))[Same*M].sum()+std.sum()
        print('loss:{:.4f} {:.4f}\n'.format(loss.item()/N,std.item()))
        
        return loss, M_sum[Same].mean().cpu(), M_sum[Diff].mean().cpu()

