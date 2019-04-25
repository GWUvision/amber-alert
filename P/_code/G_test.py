import os, torch, random, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
from torchvision import transforms
from sklearn.manifold import TSNE 
from torch import nn as nn

from .Utils import norml2, norml2Galaxy
from .Reader import ImageReader
from .color_lib import RGBmean,RGBstdv
from .resnet import resnet18, resnet50

fig = None
x = None
y = None
a = None
Labels = None
Last_conv = None

def eva(dsets, model, pb, fc_Conv):
    Labels = []
    Last_convs = []
    Fvecs = []
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=500, sampler=SequentialSampler(dsets), num_workers=16)
    torch.set_grad_enabled(False)
    for data in dataLoader:
        inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
	
        fvec = model(inputs_bt.cuda())
	last_conv = model.last_conv_out
	fmap = fc_Conv(last_conv)

        fvec = norml2Galaxy(fvec,pb[0],pb[1])
        fvec = fvec.cpu()
        Fvecs.append(fvec)
	Labels.append(labels_bt)
	Last_convs.append(fmap)

    global Labels
    Labels = torch.cat(Labels,0)
            
    global Last_conv
    Last_conv = torch.cat(Last_convs,0)

    return torch.cat(Fvecs,0)

def plot(val_tra):
    global x
    global y
    global labels
    global a
    global fig
    fig =  plt.figure()
    a = fig.add_subplot(111)
    a.set_title(val_tra + '_tsne')
    plt.scatter(x, y, c=Labels, cmap='rainbow', s=2)
    
    print('len(x): ', len(x))
    ani = animation.FuncAnimation(fig, animate, len(x), init_func=init, interval=200, repeat=True, repeat_delay=300)
    ani.save(val_tra + '_tsne_animation.gif', writer=animation.PillowWriter(fps=30))
    plt.show()

def animate(i):
    global x
    global y
    global a
    a.plot(x[:i], y[:i], linewidth=0.3, alpha=0.75)


def init():
    global x
    global y
    global Labels
    global a
    global fig
    plt.cla()
    plt.scatter(x, y, c=Labels, cmap='rainbow', s=2)
    a = fig.add_subplot(111)


def tsne(Fvecs, val_tra):
	Fvecs = Fvecs[:500,:]
	global Labels
	Labels = Labels[:500]
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(Fvecs)

	print('TSNE Results Shape: {}'.format(np.asarray(tsne_results).shape))
	
	xy_tsne = []
	xy_tsne.append(tsne_results[:,0])
	xy_tsne.append(tsne_results[:,1])
	
	plt.scatter(tsne_results[:,0], tsne_results[:,1], c=Labels, s=2, cmap='rainbow')
	#plt.title(val_tra + '_tsne')
	#plt.plot(tsne_results[:,0], tsne_results[:,1], linewidth=0.25, alpha=0.75)
	#plt.savefig(val_tra + '_tsne_results.png')
	#plt.show()	
	#plt.clf()
	global x
	global y
	x = tsne_results[:,0]
	y = tsne_results[:,1]
	plot(val_tra)
	print('Plot should be shown')

	with open('tsneResultsList.txt', 'w') as f:
		for row in xy_tsne:
			for item in row:
				f.write("%s," % item)
			f.write("\n")

def RunTest(Data, dst, data_dict, imgsize, phase, pb=[0,0],dst_model=None):
    data_transforms = transforms.Compose([transforms.Resize(imgsize),
                                          transforms.CenterCrop(imgsize),
                                          transforms.ToTensor(),
                                          transforms.Normalize(RGBmean[Data], RGBstdv[Data])])


    embed_size = pb[0]
    print(len(data_dict.items()))
    dsets = ImageReader(data_dict, data_transforms)
    if not os.path.exists(dst + 'model.pth'): 
        if not os.path.exists(dst): 
            os.makedirs(dst)
        state_dict = torch.load(dst_model + 'model.pth').state_dict()
    else:
        state_dict = torch.load(dst + 'model.pth').state_dict()

    model = resnet50().train(False) 
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, embed_size)
    model.avgpool=nn.AvgPool2d(8)

    model.load_state_dict(state_dict)
    model = model.cuda()
    ### convert fc layer to conv layer
    fc = model.fc.state_dict()
    in_ch = 2048
    out_ch = embed_size
    fc_Conv = nn.Conv2d(in_ch, out_ch, 1, 1).cuda()
    fc_Conv.load_state_dict({"weight":fc["weight"].view(out_ch, in_ch, 1, 1),
                          "bias":fc["bias"]})

    Fvecs = eva(dsets, model, pb, fc_Conv)
    #tsne(Fvecs, phase)	
    torch.save(Last_conv, dst + phase + 'Last_convs.pth')
    torch.save(Fvecs, dst + phase + 'Fvecs.pth')
    torch.save(dsets, dst + phase + 'dsets.pth')
    
    
    
    
    
