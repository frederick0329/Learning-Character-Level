require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'


local LSTM = require 'LSTM'
local Embed
if use_fuse then
    Embed = require 'Embed_fuse'
else
    Embed = require 'Embed'
end
local GRU = require 'GRU'

local model = {}

function model.buildModel(num_chars, rnn_size, embed_size, num_classes)
    local protos = {}
    protos.embed = Embed.embed(num_chars, embed_size)
    protos.gru = nn.MaskZero(GRU.gru(embed_size, rnn_size, 1), 1)
    protos.imgLookUp = nn.LookupTableMaskZero(num_chars, 3 * img_size * img_size)
    protos.criterion = nn.CrossEntropyCriterion()
    protos.classify = nn.Sequential()
    protos.classify:add(nn.Linear(rnn_size, num_classes))
    protos.softmax = nn.SoftMax()

    return protos
end

return model







