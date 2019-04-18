import torch

class_size=16

def recallAcc(Fvec, imgLabDict):
    print('Fvec: ', Fvec)
    N = len(imgLabDict)
    imgLab = torch.LongTensor([imgLabDict[i] for i in range(len(imgLabDict))])
    print('imgLab: ', imgLab)
 
    D = Fvec.mm(torch.t(Fvec))
    print('D before [eye]: ', D)
    D[torch.eye(len(imgLab)).byte()] = -1
    print('D after [eye]: ', D)
    _,idx = D.sort(1, descending=True)
    print('idx: ', idx[:,0])
    print('D after sort?: ', D)
    
    imgPre = imgLab[idx[:,0]]
    print('imgPre: ', imgPre)
    A = (imgPre==imgLab).float()
    print('A: ', A)
    for i in range(len(imgPre)):
	if A[i] < 1:
	    print('Class label: ', imgLab[i].item() * 10)
	    print('Image Label: ', i % 16)
	    print('Class predicted: ', imgPre[i].item() * 10)
	    print('Image predicted: ', idx[i,0].item() % 16)
    return (torch.sum(A)/N).item()

def RunAcc(src, phase='val'):
    dsets = torch.load(src + phase + 'dsets.pth')
    N = len(dsets)
    imgLab = dsets.idx_to_class
    Fvec = torch.load(src + phase + 'Fvecs.pth')
    print(Fvec.size())
    acc = recallAcc(Fvec, imgLab)
    print('{:.1f}'.format(acc*100))
    return acc
