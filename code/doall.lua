require 'nn'
require 'rnn'
require 'optim'
require 'cutorch'
require 'cunn'
local model = require 'model'
local Train = require 'train_gru'

cmd = torch.CmdLine()
cmd:text('Options')
-- architecture
cmd:option('-img', false, 'bool operation')
cmd:option('-fuse', false, 'bool operation')

-- input/output
cmd:option('-data_path', '', 'data path (containg .t7 files)')
cmd:option('-model_path', '', 'model path')

-- learning 
cmd:option('-gpu_id', 1, 'gpu device id')
cmd:option('-mchar', '', 'char model path (only used when -fuse is true)')
cmd:option('-mcnn', '', 'cnn model path (only used when -fuse is true)')
cmd:option('-batch_size', 400, 'batch size')
cmd:option('-max_seq_len', 10, 'sequence cutoff length')
cmd:option('rnn_size', 400, 'rnn hidden size')
cmd:option('input_size', 128, 'word embedding size')

-- others
cmd:option('sanity_check', 100, 'number of iteration to print log')
opt = cmd:parse(arg)

if opt.gpu_id >= 1 then
  cutorch.setDevice(opt.gpu_id)
  gpu = true
else
  gpu = false
end
batch_size = opt.batch_size
rho = opt.max_seq_len
rnn_size = opt.rnn_size
embed_size = opt.input_size
num_sanity_check = opt.sanity_check


num_epochs = 20
num_classes = 12
img_size = 36
use_img = opt.img
use_fuse = opt.fuse
num_chars = 0
for f in paths.files(opt.data_path..'/img/') do
  if f ~= '.' and f ~= '..' then
    num_chars = num_chars+1
  end
end

local sgd_params = {
    learningRate = 1e-3,
}


model_prefix = opt.model_path

if not paths.dir(model_prefix) then
    paths.mkdir(model_prefix)
end

local model = model.buildModel(num_chars, rnn_size, embed_size, num_classes)
if use_fuse then
    char_pretrained_model = torch.load(opt.mchar)
    cnn_pretrained_model = torch.load(opt.mcnn)
end

if opt.img then
  local imgs = torch.load(opt.data_path..'/img_36.t7')
  model.imgLookUp.weight[{{1,num_chars},{}}] = imgs:view(num_chars,  3 * img_size * img_size)
end

if gpu then
  for k, v in pairs(model) do 
      v = v:cuda()
  end
end

print('Training...')
local train_path = opt.data_path..'/train.t7'
local val_path = opt.data_path..'/val.t7'
local ds_train = torch.load(train_path)
local ds_val = torch.load(val_path)
Train.train_sgd(model, ds_train, ds_val, sgd_params)

