# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""

import torch
import torch.nn as nn
import torch.nn.parallel

class D(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dfs = 5, ngpu = 1):
        super(D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(

            nn.Conv2d(nc, ndf, dfs, 2, dfs//2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf),
           
            nn.Conv2d(ndf, ndf*2, dfs, 2, dfs//2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf*2),
           
            nn.Conv2d(ndf*2, ndf*4, dfs, 2, dfs//2, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf*4),
          
            nn.Conv2d(ndf*4, ndf*8, dfs, 2, dfs//2, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf*8),

            nn.Conv2d(ndf * 8, 1, kernel_size=dfs, stride=2, padding=2, bias=False)
            
        )
        self.main = main
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean()
        
        return output.view(1)
    
class G(nn.Module):
    def __init__(self, nc = 1, nz = 1, ngf = 64, gfs = 5, ngpu = 1):
        super(G, self).__init__()
        self.ngpu = ngpu
   
        if gfs < 5:
            out_pad=1
        else:
            out_pad=0

        self.main = nn.Sequential(
             
                nn.ConvTranspose2d(     nz, ngf * 8, gfs, 2, gfs//2, bias=False), 
                nn.ReLU(True),
                nn.BatchNorm2d(ngf * 8),
               
                nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(ngf * 4),
                
                nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(ngf * 2),
                
                nn.ConvTranspose2d(ngf * 2,     ngf, gfs, 2, gfs//2, out_pad,bias=False),
                nn.ReLU(True),
                nn.BatchNorm2d(ngf),
                
                nn.ConvTranspose2d(    ngf,      nc, gfs, 2, 2, bias=False),
                nn.Tanh()
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output
    
