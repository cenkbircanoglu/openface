--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 14.01.2017
-- Time: 09:49
-- To change this template use File | Settings | File Templates.
--

imgDim = 224

function createModel()

    local SpatialConvolution = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling

    net:add(SpatialConvolution(3, 64, 11, 11, 4, 4, 2, 2)) -- 224 -> 55
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(SpatialMaxPooling(3, 3, 2, 2)) -- 55 ->  27
    net:add(SpatialConvolution(64, 192, 5, 5, 1, 1, 2, 2)) --  27 -> 27
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(192))
    net:add(SpatialMaxPooling(3, 3, 2, 2)) --  27 ->  13
    net:add(SpatialConvolution(192, 384, 3, 3, 1, 1, 1, 1)) --  13 ->  13
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(384))
    net:add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)) --  13 ->  13
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) --  13 ->  13
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(SpatialMaxPooling(3, 3, 2, 2)) --  27 ->  13
    net:add(nn.View(256 * 6 * 6)) --Changed
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(256 * 6 * 6, 4096)) --Changed
    net:add(nn.ReLU(true))
    net:add(nn.BatchNormalization(4096))
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(4096, 4096))
    net:add(nn.ReLU(true))
    net:add(nn.BatchNormalization(4096))

    net:add(nn.Linear(4096, opt.embSize))
    net:add(nn.Normalize(2))

    return net
end


