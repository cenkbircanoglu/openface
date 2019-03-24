--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 04.06.2018
-- Time: 20:42
-- To change this template use File | Settings | File Templates.
--


imgDim = 64




function createModel()

    local SpatialConvolution = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling

    local net = nn.Sequential()
    net:add(SpatialConvolution(3, 6, 5, 5, 1, 1, 0, 0))
    net:add(nn.ReLU(true))
    net:add(SpatialMaxPooling(2, 2))
    net:add(SpatialConvolution(6, 16, 5, 5, 1, 1, 0, 0))
    net:add(nn.ReLU(true))
    net:add(SpatialMaxPooling(2, 2))
    net:add(nn.View(16 * 13 * 13))
    net:add(nn.Linear(16 * 13 * 13, 512))
    net:add(nn.ReLU(true))
    net:add(nn.Linear(512, 256))
    net:add(nn.ReLU(true))
    net:add(nn.Linear(256, opt.embSize))
    net:add(nn.Normalize(2))

    return net
end




