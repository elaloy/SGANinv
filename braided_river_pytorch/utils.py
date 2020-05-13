# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""
import os
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import FLIP_LEFT_RIGHT
import h5py

def image_to_tensor(img):
    '''
    convert image to Theano/Lasagne 3-tensor format;
    changes channel dimension to be in the first position and rescales from [0,255] to [-1,1]
    '''
#    tensor = np.array(img).transpose( (2,0,1) )
#    tensor = tensor / 128. - 1.
    i_array=np.array(img)
    if len(i_array.shape)==2: # the array is 2d, convert to a 3D array
        i_array=i_array.reshape((i_array.shape[0],i_array.shape[1],1))
    tensor = i_array.transpose( (2,0,1) )
    tensor = tensor / 128. - 1.0
    return tensor


def tensor_to_2Dimage(tensor):
    '''
    convert 3-tensor to image;
    changes channel to be last and rescales from [-1, 1] to [0, 255]
    '''
    img = np.array(tensor).transpose( (1,2,0) )
    img = (img + 1.) * 128.
    return np.uint8(img)
    

def get_texture2D_iter(folder, npx=128, npy=128,batch_size=64, \
                     filter=None, mirror=True, n_channel=1):
    '''
    @param folder       iterate of pictures from this folder
    @param npx          size of patches to extract
    @param n_batches    number of batches to yield - if None, it yields forever
    @param mirror       if True the images get augmented by left-right mirroring
    @return a batch of image patches fo size npx x npx, with values in [-1,1]
    '''
    HW1    = npx
    HW2    = npy
    imTex = []
    files = os.listdir(folder)
    for f in files:
        name = folder + f
        try:
            img = Image.open(name)
            imTex += [image_to_tensor(img)]
            if mirror:
                img = img.transpose(FLIP_LEFT_RIGHT)
                imTex += [image_to_tensor(img)]
        except:
            print("Image ", name, " failed to load!")

    while True:
        data=np.zeros((batch_size,n_channel,npx,npx))                   # NOTE: assumes n_channel channels!
        for i in range(batch_size):
            ir = np.random.randint(len(imTex))
            imgBig = imTex[ir]
            if HW1 < imgBig.shape[1] and HW2 < imgBig.shape[2]:   # sample patches
                h = np.random.randint(imgBig.shape[1] - HW1)
                w = np.random.randint(imgBig.shape[2] - HW2)
                img = imgBig[:, h:h + HW1, w:w + HW2]
            else:                                               # whole input texture
                img = imgBig
            data[i] = img

        yield data
        
def get_texture2D_samples(folder,batch_size=64):
    '''
   still need to document this one but here we get the batches from a given
   set of training images that already have the required npx x npy  size
    '''
    imTex = []
    files = os.listdir(folder)
    for f in files:
        name = folder + f
        try:
            with h5py.File(name, 'r') as fid:
                img=np.array(fid['features']) # the 2D training images are already under (channel, W, H, D) format and [0,1] scale
			# img needs to be rescaled from [0,1] to [-1,1]
            img=((img*1.0)*2-1).astype('float32')
            imTex += [img]
            print("Image ", name, " loaded")

        except:
            print("Image ", name, " failed to load!")
       
    while True:
        
        ir = 0
        ivec=np.random.choice(np.arange(0,img.shape[0]), size=batch_size, replace=True)
        data=imTex[ir][ivec]
        yield data

        


def save_tensor2D(tensor, filename):
    '''
    save a 3-tensor (channel, x, y) to image file
    '''
    img = tensor_to_2Dimage(tensor)
    
    if img.shape[2]==1:
        # print('not now either!')
        img=img.reshape((img.shape[0],img.shape[1]))
        img = Image.fromarray(img).convert('L')
    else:
        img = Image.fromarray(img)
        
    img.save(filename)
    
    
def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    #return zx*2**depth
    return (zx-1)*2**depth + 1
    
if __name__=="__main__":
    print("nothing here.")
