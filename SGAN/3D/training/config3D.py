import os
from tools import create_dir
from data_io3D import get_texture_iter


create_dir('samples')               # create, if necessary, for the output samples 
create_dir('models') 


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return (zx - 1)*2**depth + 1


class Config(object):
    '''
    wraps all configuration parameters in 'static' variables
    '''
    ##
    # network parameters
    nz          = 1                  # num of dim for Z at each field position 
    zx          = 4                    # number of spatial dimensions in Z 
    zx_sample   = 5                     # size of the spatial dimension in Z for producing the samples
	# l = h/r, m = w/r
    nc          = 1                     # number of channels in input X (i.e. r,g,b)
    n_conv_layer = 5 
    # kernel sizes on each layer - should be odd numbers for zero-padding stuff:       
    gen_ks      = ([(5,5,5)] * n_conv_layer)[::-1]
    dis_ks      = [(5,5,5)] * n_conv_layer   
    gen_ls      = len(gen_ks)           # num of layers in the generative network
    dis_ls      = len(dis_ks)           # num of layers in the discriminative network
    gen_fn      = [nc]+[2**(n+6) for n in range(gen_ls-1)] # generative number of filters
    gen_fn      = gen_fn[::-1]
    dis_fn      = [2**(n+6) for n in range(dis_ls-1)]+[1] # discriminative number of filters

    lr          = 0.0005                # learning rate of adam
    b1          = 0.5                   # momentum term of adam
    l2_fac      = 1e-5                  # L2 weight regularization factor

    batch_size  = 25 

    epoch_iters = batch_size * 100
    num_epoch=50

    k           = 1                     # number of D updates vs G updates

    npx         = zx_to_npx(zx, gen_ls) # num of pixels width/height/depth of 3D images in X

    ##
    # data input folder
    sub_name    = 'ti_3D' # 3D training image
    home        = os.path.expanduser("~")
    texture_dir = home + "/SGANinv/SGAN/3D/training/%s/" % sub_name
    data_iter   = get_texture_iter(texture_dir, npx=npx, mirror=False, batch_size=batch_size,n_channel=nc)

    save_name   = sub_name+ "_filters%d_npx%d_%dgL_%ddL" % (dis_fn[0],npx,gen_ls, dis_ls)

    load_name   = None                  # if None, initializing network from scratch
    #load_name   = "fold3Dcat_filters64_npx97_5gL_5dL_epoch16.sgan"


    @classmethod
    def print_info(cls):
        ##
        # output some information
        print("Learning and generating samples from zx ", cls.zx, ", which yields images of size npx ", zx_to_npx(cls.zx, cls.gen_ls)) 
        print("Producing samples from zx_sample ", cls.zx_sample, ", which yields images of size npx ", zx_to_npx(cls.zx_sample, cls.gen_ls)) 
        print("Saving samples and model data to file ", cls.save_name)

