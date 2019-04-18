import os, time, copy, random
from glob import glob

from torchvision import models, transforms, datasets
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn as nn
import torch

from .Sampler import BalanceSampler
from .Reader import ImageReader
from .Loss import NpairsLossPB,NpairsLoss
from .G_eval import recallAcc
from .G_test import eva
from .color_lib import RGBmean,RGBstdv

PHASE = ['tra','val']

import numpy as np
from sklearn.decomposition import PCA
from .Utils import norml2
import matplotlib.pyplot as plt
def projection(pca,Y,n):
    X = Y# - pca.mean_
    D = norml2(torch.from_numpy(pca.components_)).numpy()
    X_transformed = np.dot(X, D[:n].T)
#     X_transformed /= np.sqrt(pca.explained_variance_)
    return X_transformed


class learn():
    def __init__(self, gpuid, dst, Data, data_dict, batch_size=128, model_name='r18', init_lr=0.001, decay=0.1, imgsize=256, avg=8, num_workers=16):
        self.dst = dst
        self.gpuid = gpuid
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.decay_time = [False,False]
        self.init_lr = init_lr
        self.decay_rate = decay
        self.model_name = model_name
        self.avg = avg
        
        self.Data = Data
        self.data_dict = data_dict
        self.imgsize = imgsize
        self.RGBmean = RGBmean[Data]
        self.RGBstdv = RGBstdv[Data]
        
        if not self.setsys(): print('system error'); return
        
    def runNpairsPB(self, emb_dim, th, pb, sigma=0.1, num_epochs=20):
        self.out_dim = emb_dim
        self.num_epochs = num_epochs
        self.pb = pb
        self.loadData()
        self.setModel()
        self.criterion = NpairsLossPB(sigma,th,pb) 
        
        print('output dimension: {}'.format(emb_dim))
        print('pb parameters: {}'.format(pb))
        self.opt()

    def runNpairs(self, emb_dim, sigma=0.1, num_epochs=20):
        self.out_dim = emb_dim
        self.num_epochs = num_epochs
        self.pb = [emb_dim,1]
        self.loadData()
        self.setModel()
        self.criterion = NpairsLoss(sigma)
        
        print('output dimension: {}'.format(emb_dim))
        self.opt()
        
    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        self.tra_transforms = transforms.Compose([transforms.Resize(int(self.imgsize*1.1)),
                                                  transforms.RandomCrop(self.imgsize),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])
        
        self.val_transforms = transforms.Compose([transforms.Resize(self.imgsize),
                                                  transforms.CenterCrop(self.imgsize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])

        self.dsets = ImageReader(self.data_dict['tra'], self.tra_transforms) 
        self.intervals = self.dsets.intervals
        self.classSize = len(self.intervals)
        print('number of classes: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        if self.model_name == 'r18':
            self.model = models.resnet18(pretrained=True)
            print('Setting model: resnet18')
        else:
            self.model = models.resnet50(pretrained=True)
            print('Setting model: resnet50')
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.out_dim)
        self.model.avgpool=nn.AvgPool2d(self.avg)

        print('Training on Single-GPU')
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.0)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.6*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.9*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self):
        self.model.train(True)  # Set model to training mode

        dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, sampler=BalanceSampler(self.intervals, GSize=16), num_workers=self.num_workers)
        
        L_data, N_data = 0.0, 0
        R_same, R_diff, bt = 0.0, 0.0, 0
        # iterate batch
        L = []
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = self.model(inputs_bt.cuda())
                loss, rt_same, rt_diff = self.criterion(fvec, labels_bt.cuda())

                loss.backward()
                self.optimizer.step()  
            
            L_data += loss.item()
            N_data += len(labels_bt)

            R_same += rt_same
            R_diff += rt_diff
            bt+=1
            L.append(loss.item()/len(labels_bt))
            
        return L_data/N_data, R_same/bt, R_diff/bt, L
        
    def printAccs(self,epoch):
        self.model.train(False)  # Set model to training mode
        dsets_tra = ImageReader(self.data_dict['tra'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        print('Calling eva()')
        Fvec_tra = eva(dsets_tra, self.model, self.pb)
        Fvec_val = eva(dsets_val, self.model, self.pb)
        print('Done eva()')
        
        acc_tra = recallAcc(Fvec_tra, dsets_tra.idx_to_class)
        acc_val = recallAcc(Fvec_val, dsets_val.idx_to_class)

	return acc_tra, acc_val

    def recall(self,epoch):
        self.model.train(False)  # Set model to training mode
        dsets_tra = ImageReader(self.data_dict['tra'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        print('Calling eva()')
        Fvec_tra = eva(dsets_tra, self.model, self.pb)
        Fvec_val = eva(dsets_val, self.model, self.pb)
        print('Done eva()')
        
        acc_tra = recallAcc(Fvec_tra, dsets_tra.idx_to_class)
        acc_val = recallAcc(Fvec_val, dsets_val.idx_to_class)
        
        valimgCat = dsets_val.idx_to_class
        traimgCat = dsets_tra.idx_to_class

        pca = PCA(n_components=self.out_dim)
        pca.fit(Fvec_tra.numpy())

        plt.figure()
        plt.plot(pca.singular_values_)
        plt.xlim(0,self.out_dim)
        plt.ylim(0,40)
        plt.title('tra-singular-values')
        print('Saving singular values')
        plt.savefig(self.dst+'{:02}-tra-singular-values'.format(epoch))
        print('Saving singular values done')
        
        plt.figure()
	print('Computing projection train')
        Y_tra = projection(pca,Fvec_tra.numpy(),self.out_dim)
	print('Computing projection train done')
        A = []
        for i in range(4,self.out_dim,4):
	    print('Computing recallAcc on tra: ', i)
            acc_pca = recallAcc(norml2(torch.from_numpy(Y_tra[:,:i])),traimgCat)
	    print('Computing recallAcc on tra done')
            A.append(acc_pca)
        plt.plot(range(4,self.out_dim,4),A,label='tra-comp-tra-recall')

	print('Computing projection val')
        Y_val = projection(pca,Fvec_val.numpy(),self.out_dim)
	print('Computing projection val done')
        A = []
        for i in range(4,self.out_dim,4):
            acc_pca = recallAcc(norml2(torch.from_numpy(Y_val[:,:i])),valimgCat)
            A.append(acc_pca)
        plt.plot(range(4,self.out_dim,4),A,label='tra-comp-val-recall')

	print('Computing PCA')
        pca = PCA(n_components=self.out_dim)
        pca.fit(Fvec_val.numpy())
	print('Computing PCA done')
	print('Computing projection again with new PCA? val')
        Y_val = projection(pca,Fvec_val.numpy(),self.out_dim)
	print('Computing projection again with new PCA? val done')
        A = []
        for i in range(4,self.out_dim,4):
            acc_pca = recallAcc(norml2(torch.from_numpy(Y_val[:,:i])),valimgCat)
            A.append(acc_pca)

        plt.plot(range(4,self.out_dim,4),A,label='val-comp-val-recall')
        plt.title('dim-recall')
        plt.legend()
        plt.xlim(0,self.out_dim)
        print('Saving dim recall')
        plt.savefig(self.dst+'{:02}-dim-recall'.format(epoch))
        print('Saving dim recall done')
        return acc_tra, acc_val
    
    def opt(self):
        # recording time and epoch info
        since = time.time()
        self.record = []
        for epoch in range(self.num_epochs): 
            # adjust the learning rate
            print('Epoch {}/{} \n '.format(epoch+1, self.num_epochs) + '-' * 40)
            self.lr_scheduler(epoch)
            
            # train 
            print('Training Begin')
            tra_loss, maskRt_same, maskRt_diff, _ = self.tra()
            print('Training Done')            

            # save model for each epoch
            print('Saving model')
            torch.save(self.model, self.dst + 'model.pth')
            print('Saving model compilete')
            
            # calculate the retrieval accuracy
            print('Computing recall')
            acc_tra, acc_val = self.printAccs(epoch)
            print('Computing recall complete')
            print('Loss: {:.3f}  R@1_tra:{:.1f}  R@1_val:{:.1f}'.format(tra_loss, acc_tra*100, acc_val*100)) 
            self.record.append((epoch, tra_loss, maskRt_same, maskRt_diff))

            
        torch.save(torch.Tensor(self.record), self.dst + 'record.pth')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        return
    
    

    
