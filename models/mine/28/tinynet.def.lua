--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 04.06.2018
-- Time: 20:42
-- To change this template use File | Settings | File Templates.
--


imgDim = 28




function createModel()

    local SpatialConvolution = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling

    local net = nn.Sequential()
    net:add(SpatialConvolution(3, 10, 5, 5, 1, 1, 0, 0))
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
    net:add(nn.Linear(50, opt.embSize))
    net:add(nn.Normalize(2))

    return net
end




