import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def invDict(dict_in):
    """input type dict"""
    values = sorted(set([v for k,v in dict_in.items()]))
    dict_out = {v:[] for v in values}
    for k,v in dict_in.items(): dict_out[v].append(k)
    return dict_out

def norml2(vec):# input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
    return vec.div(w)

def norml2Galaxy(vec, d=0, l=0):# input N by F
    if l==1:
        return norml2(vec)
    else:
        N = vec.size(0)
        F = vec.size(1)
        if d*l!=F: 
            print('dimension error')
            return
        V = vec.view(N*l,d)
        w = torch.sqrt((torch.t(V.pow(2).sum(1).repeat(d,1))))
        return (V.div(w)).view(N,F)

def Mat(Lvec):
    N = Lvec.size(0)
    M = torch.zeros(N,N)
    I = torch.eye(N)
    
    for l in list(set(Lvec.data.tolist())):
        v = (Lvec==l).view(-1,1).data.float()
        M += (v.mm(torch.t(v)))
        
    return (M-I).byte(), (1-M).byte()#same diff

def distributionPlot(src, norm=1, savefigname=None):
    traFvec = torch.load(src + 'traFvecs.pth')
    valFvec = torch.load(src + 'valFvecs.pth')

    tradset = torch.load(src + 'tradsets.pth')
    valdset = torch.load(src + 'valdsets.pth')

    traLab = torch.LongTensor([tradset.idx_to_class[i] for i in range(len(tradset.idx_to_class))])
    valLab = torch.LongTensor([valdset.idx_to_class[i] for i in range(len(valdset.idx_to_class))])

    intv = 101

    # tra dset
    # matting
    S,D = Mat(traLab)
    M = (traFvec.mm(torch.t(traFvec))/norm).cpu()
    M[torch.eye(M.size(0)).byte()]=-norm

    H_tra_same = torch.histc(M[S], bins=intv, min=-1, max=1)/M[S].size(0)
    H_tra_diff = torch.histc(M[D], bins=intv, min=-1, max=1)/M[D].size(0)

    # val dset    
    # matting
    S,D = Mat(valLab)
    M = (valFvec.mm(torch.t(valFvec))/norm).cpu()
    M[torch.eye(M.size(0)).byte()]=-norm

    H_val_same = torch.histc(M[S], bins=intv, min=-1, max=1)/M[S].size(0)
    H_val_diff = torch.histc(M[D], bins=intv, min=-1, max=1)/M[D].size(0)

    plt.figure()
    plt.plot(np.linspace(-1,1,intv),H_tra_diff.numpy(),'r:',label='tra_diff')
    plt.plot(np.linspace(-1,1,intv),H_val_diff.numpy(),'b:',label='val_diff')
    plt.plot(np.linspace(-1,1,intv),H_tra_same.numpy(),'r-',label='tra_same')
    plt.plot(np.linspace(-1,1,intv),H_val_same.numpy(),'b-',label='val_same')
    plt.xlim(-1,1)
    plt.ylim(0,0.4)
    
    # dim = traFvec.size(1)
    # print(dim)
    # a = norml2(torch.randn(1000,dim))
    # b = norml2(torch.randn(1000,dim))
    # D = a.mm(b.t())
    # H = torch.histc(D, bins=101, min=-1, max=1)
    # H = H/H.sum()
    # plt.plot(torch.linspace(-1,1,101).tolist(),H.numpy(),label='rand_dim='+str(dim))
    
    plt.legend()
    plt.xlabel('dot product')
    plt.ylabel('% of samples')
    
    if savefigname!=None:
        plt.savefig(src+savefigname)
        
    return H_tra_same, H_tra_diff, H_val_same, H_val_diff

    # M_diff = M.clone()
    # M_diff[S] = -norm
    # d_val_diff = M_diff.topk(TK)[0].view(-1)
    # d_val_diff = M_same[D]