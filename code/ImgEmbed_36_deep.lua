require 'nngraph'
require 'nn'

local ImgEmbed = {}
    -- a convolutional neuralnet
function ImgEmbed.embed(embed_size)
    local inputs = {}
    table.insert(inputs, nn.Identity()())   -- network input
    local img_inputs = inputs[1]
    local img = nn.View(-1, 3, 36, 36):setNumInputDims(2)(img_inputs)

    local conv1 = nn.SpatialConvolution(3, 32, 3, 3)(img)
    local activation1 = nn.ReLU()(conv1)
    local max_pool1 = nn.SpatialMaxPooling(2, 2)(activation1)

    local conv2 = nn.SpatialConvolution(32, 32, 3, 3)(max_pool1)
    local activation2 = nn.ReLU()(conv2)
    local max_pool2 = nn.SpatialMaxPooling(2, 2)(activation2)

    local conv3 = nn.SpatialConvolution(32, 32, 3, 3)(max_pool2)
    local activation3 = nn.ReLU()(conv3)
    local flatten = nn.View(32*5*5):setNumInputDims(3)(activation3)

    local fc1 = nn.Linear(32*5*5, 128)(flatten)
    local fc1_activation = nn.ReLU()(fc1)
    local fc2 = nn.Linear(128, 128)(fc1_activation)
    local img_embed = nn.ReLU()(fc2)

    outputs = {}
    table.insert(outputs, img_embed)
    return nn.gModule(inputs, outputs)
end

return ImgEmbed
