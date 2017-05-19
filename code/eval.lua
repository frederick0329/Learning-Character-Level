require 'nn'
require 'cutorch'
require 'rnn'
require 'optim'
require 'cunn'
require 'nngraph'
local model_utils = require 'model_utils'
local Train = require 'train_gru'


cmd = torch.CmdLine()
cmd:text('Options')
-- architecture
cmd:option('-img', false, 'bool operation')
cmd:option('-fuse', false, 'bool operation')
cmd:option('-rnn_size', 400, 'rnn hidden size')

-- data/model
cmd:option('-data_path', '', 'test data path (.t7 file)')
cmd:option('-model_path', '', 'model path')

-- others
cmd:option('-batch_size', 500, 'batch size')
cmd:option('-rho', 10, 'cut off length')
cmd:option('-gpu_id', 1, 'gpu device id')
opt = cmd:parse(arg)

if opt.gpu_id >= 1 then
  cutorch.setDevice(opt.gpu_id)
  gpu = true
end

batch_size = opt.batch_size
rho = opt.rho
rnn_size = opt.rnn_size

img_size = 36
use_img = opt.img
use_fuse = opt.fuse
num_chars = 0
for f in paths.files(opt.data_path..'/img/') do
  if f ~= '.' and f ~= '..' then
    num_chars = num_chars+1
  end
end

local ds = torch.load(opt.data_path..'/test.t7')
local size = ds.input:size(1)
local model = torch.load(opt.model_path)

local img_path = opt.data_path..'/img_36.t7'
local imgs = torch.load(img_path)
local img_size = 36
model.imgLookUp.weight = torch.reshape(imgs, num_chars, 3*36*36)
if gpu == true then
  model.imgLookUp.weight = model.imgLookUp.weight:cuda()--imgs:view(num_chars,  3 * 36 * 36)
end


clones = {}
clones.model = {}
clones.model.gru = model_utils.clone_many_times(model.gru, rho, not model.gru.parameters)
clones.model.embed = model_utils.clone_many_times(model.embed, rho, not model.embed.parameters)
clones.model.imgLookUp = model.imgLookUp

num_batches = math.ceil(size/batch_size)

--[[
out = {}
out['res'] = torch.Tensor(ds.input:size(1))
out['input'] = torch.Tensor(ds.input:size())
--]]

local loss = 0
local precision = 0
for n = 1, num_batches do 
    local inputs = ds.input:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, size-(n-1)*batch_size)):clone():double()
    local masks
    if use_img then
        masks = ds.mask:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, size-(n-1)*batch_size)):clone():double()
    else
        masks = torch.ones(inputs:size())
    end
    local targets = ds.target:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, size-(n-1)*batch_size)):clone():double()
    if gpu then
	    inputs = inputs:cuda()
        masks = masks:cuda()
        targets = targets:cuda()
    end
  	local prob
    local err
    local val
    local outputs = {}
   	prob, err = Train.forward(clones, model, inputs, masks, targets, outputs)
    loss = loss + err
    val, id = torch.max(prob, 2)
    if gpu then
        id = id:cuda()
    end
    -- out['res'][{{1+(n-1)*batch_size, (n-1)*batch_size+math.min(batch_size, size-(n-1)*batch_size)}}] = targets:eq(id):double()
    -- out['input'][{{1+(n-1)*batch_size, (n-1)*batch_size+math.min(batch_size, size-(n-1)*batch_size)}, {}}] = inputs:double()
   	precision = precision + torch.sum(targets:eq(id)) / math.min(batch_size, size-(n-1)*batch_size)
end
print(string.format("test loss = %f ", loss/(num_batches)))
print(string.format("precision@1 = %f ", precision/(num_batches)))

--[[
if train_lang == test_lang then
    torch.save(output_path,out)
end
--]]
