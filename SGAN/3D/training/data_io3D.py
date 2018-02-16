import os
import numpy as np
from PIL import Image

import h5py

def tensor_to_2Dimage(tensor):
    '''
    convert 3-tensor to image;
    changes channel to be last and rescales from [-1, 1] to [0, 255]
    '''
    img = np.array(tensor).transpose( (1,2,0) )
    img = (img + 1.) * 128.
    return np.uint8(img)
    

def get_texture_iter(folder, npx=128, batch_size=64, \
                     filter=None, mirror=True, n_channel=3):
    '''
    @param folder       iterate of pictures from this folder
    @param npx          size of patches to extract
    @param n_batches    number of batches to yield - if None, it yields forever
    @param mirror       if True the images get augmented by left-right mirroring
    @return a batch of image patches fo size npx x npx x npx, with values in [0,1]
    '''
    HW    = npx
    imTex = []
    files = os.listdir(folder)
    for f in files:
        name = folder + f
        try:
            with h5py.File(name, 'r') as fid:
                img=np.array(fid['features']) # the 3D training image is already under (channel, W, H, D) format and [0,1] scale
			# img needs to be rescaled from [0,1] to [-1,1]
            img=((img*1.0)*2-1).astype('float32')
            imTex += [img]
            if mirror:
                pass
    			# Mirror not yet implemented for 3D
        except:
            print("Image ", name, " failed to load!")

    while True:
        data=np.zeros((batch_size,n_channel,npx,npx,npx))                   # NOTE: assumes n_channel channels!
        for i in range(batch_size):
            ir = np.random.randint(len(imTex))
            imgBig = imTex[ir]
            if HW < imgBig.shape[1] and HW < imgBig.shape[2] and HW < imgBig.shape[3]:   # sample patches
                h = np.random.randint(imgBig.shape[1] - HW)
                w = np.random.randint(imgBig.shape[2] - HW)
                d = np.random.randint(imgBig.shape[3] - HW)
                img = imgBig[:, h:h + HW, w:w + HW, d:d + HW]
            else:                                               # whole input texture
                img = imgBig
            data[i] = img

        yield data


def save_tensor(tensor, filename):
    '''
    save a 4-tensor (channel, x, y, z) in [-1,1] to a hdf5 file in [0,1]
    '''
    # Save a 2D slice in X-Z
    n=np.round(tensor.shape[1]*0.5).astype('int')
    img = tensor_to_2Dimage(tensor[:,n,:,:])
    if img.shape[2]==1:
        img=img.reshape((img.shape[0],img.shape[1]))
        img = Image.fromarray(img).convert('L')
    else:
        img = Image.fromarray(img)
    img.save(filename+'.png')
	
     # save 4D tensor
    tensor=(tensor+1)*0.5 # Convert from [-1,1] to [0,1]
    f = h5py.File(filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=tensor)
	# Close HDF5 file
    f.flush()
    f.close()


if __name__=="__main__":
    print("nothing here.")
