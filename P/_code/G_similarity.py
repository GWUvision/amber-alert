import torch
import os
import numpy as np
from .similarity_ops import compute_spatial_similarity 
from .image_ops import *

class_size=16

def individualSimilarity(Fvec, clazz, i1, i2):    
    # Calculate the index of the Fvec we care about
    fvec_i1 = clazz//10 * class_size + i1
    Fvec1 = Fvec[fvec_i1].numpy()
    print('Fvec1 before reshape: ', Fvec1)
    # Convert shape (64,) to (64, 1)
    Fvec1 = Fvec1.reshape(Fvec1.shape[-1], -1)

    fvec_i2 = clazz//10 * class_size + i2
    Fvec2 = Fvec[fvec_i2].numpy()
    Fvec2 = Fvec2.reshape(Fvec2.shape[-1], -1)
    print('Fvec1 type: ', type(Fvec1))
    print('Fvec1 shape: ', np.shape(Fvec1))
    print('Fvec[{}]: {}'.format(fvec_i1, Fvec1))
    print('Fvec[{}]: {}'.format(fvec_i2, Fvec2))

    # Compute heatmaps
    heatmap1, heatmap2 = compute_spatial_similarity(Fvec1, Fvec2)

    print('heatmap1: ', heatmap1)
    print('heatmap2: ', heatmap2)
    
    # Load the image + superimose heatmap
    zfilled_i1 = str(i1).zfill(2)
    zfilled_i2 = str(i2).zfill(2)
    im1_with_similarity = combine_image_and_heatmap(load_and_resize('images/val/'+str(clazz)+'/'+zfilled_i1+'.png'),heatmap1)
    im2_with_similarity = combine_image_and_heatmap(load_and_resize('images/val/'+str(clazz)+'/'+zfilled_i2+'.png'),heatmap2)

    # Save image
    combined_image = pil_bgr_to_rgb(combine_horz([im1_with_similarity,im2_with_similarity]))
    combined_image.save(os.path.join('./_result/test_similarity.jpg'))
    return

def RunSimilarity(src, phase='val'):
    # Load Fvecs (output high-dim vector) of all images
    Fvec = torch.load(src + phase + 'Fvecs.pth')
    print(Fvec.size())
    # Pass Fvecs, class, image index 1, image index 2
    individualSimilarity(Fvec, 1000, 0, 10)
