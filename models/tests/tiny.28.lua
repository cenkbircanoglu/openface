--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 11.01.2017
-- Time: 19:41
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'torch'
require 'dpnn'
require 'optim'
require 'image'
require 'torchx'
require 'optim'
require 'xlua'

paths.dofile('../../training/torch-TripletEmbedding/TripletEmbedding.lua')
torch.setdefaulttensortype("torch.FloatTensor")
a = torch.rand(10, 1, 28, 28)


function createModel()

    local SpatialConvolution = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling

    local net = nn.Sequential()
    net:add(SpatialConvolution(1, 10, 5, 5, 1, 1, 0, 0))
    net:add(SpatialMaxPooling(2, 2))
    net:add(nn.ReLU(true))
    net:add(SpatialConvolution(10, 20, 5, 5, 1, 1, 0, 0))
    net:add(nn.Dropout(0.5))
    net:add(SpatialMaxPooling(2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.View(320))
    net:add(nn.Linear(320, 50))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(50, 128))
    net:add(nn.Normalize(2))

    return net
end

net = createModel()
print(net:forward(a):size())


