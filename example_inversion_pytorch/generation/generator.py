"""
Created on Sat May 19 10:04:09 2018
@author: elaloy <elaloy elaloy@sckcen.be>

"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, gpath, cuda=False):
        super(Generator, self).__init__()
        nc = 1
        nz = 1
        ngf = 64
        gfs = 5
        self.main = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, gfs, 2, gfs//2, bias=False),
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, 2, gfs//2, bias=False),
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, 2, gfs//2, bias=False),
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ConvTranspose2d(ngf * 2,     ngf, gfs, 2, gfs//2, bias=False),
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf),
            nn.ConvTranspose2d(    ngf,      nc, gfs, 2, 2, bias=False),
            nn.ReLU(True),
            # Do some dilations #
            nn.ConvTranspose2d(     nc, ngf, gfs, 1, 6, output_padding=0,bias=False,dilation=3), 
            nn.ReLU(True),
            nn.InstanceNorm2d(ngf),
            nn.ConvTranspose2d(    ngf,  nc, gfs, 1, 10, output_padding=0, bias=False,dilation=5),
            nn.Tanh()
        )
        if cuda:
            self.load_state_dict(torch.load(gpath))
        else:
            self.load_state_dict(torch.load(gpath,
                                            map_location=lambda storage,
                                            loc: storage))

    def forward(self, input):
        output = self.main(input)
        return output
