require 'io'
require 'nn'
require 'math'

function string:split( inSplitPattern, outResults )
 
   if not outResults then
      outResults = {}
   end
   local theStart = 1
   local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
   while theSplitStart do
      table.insert( outResults, string.sub( self, theStart, theSplitStart-1 ) )
      theStart = theSplitEnd + 1
      theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
   end
   table.insert( outResults, string.sub( self, theStart ) )
   return outResults
end

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-n', 100)
cmd:option('-r', 10)
cmd:option('-c', 12)
cmd:option('-s', 'train')
cmd:option('-l', 'zh_traditional')
opt = cmd:parse(arg)
lang = opt.l
nInst = opt.n
rho = opt.r
nTarget = opt.c
split = opt.s

--split = 'train'
--lang = 'zh_traditional'


filePath = './' ..lang .. '/' .. lang .. '_' .. split .. '.txt'
--outPath = 'train_100.t7'
file = io.open(filePath)
--nTarget = 12
--nInst = 573073
--rho = 10

dataset = {}
dataset.input = torch.ShortTensor(nInst, rho):zero()
dataset.mask = torch.ByteTensor(nInst, rho):zero()
dataset.target = torch.ShortTensor(nInst):zero()

insCount = 1

local shuffle = torch.randperm(nInst)

for line in file:lines() do
    local i = 1
    local part = string.split(line, '\t')
    for iPart = 1,#part do
        local token = part[iPart]
        local tmp = string.split(token, ',')
        if i == 1 then
            dataset.target[insCount] = tonumber(token)
            i = i + 1
        elseif i == 2 then
            local sent = torch.ShortTensor(1, rho):zero()
            local idx = math.max(rho-#tmp+1, 1)
            for j = 1,#tmp do
                sent[{1,idx}] = tonumber(tmp[j])
                idx = idx + 1
                if idx > rho then
                    break
                end
            end
            dataset.input[{insCount, {}}] = sent
            i = i + 1
        else 
            local mask = torch.ByteTensor(1, rho):zero()
            local idx = math.max(rho-#token+1, 1)
            for j = 1,#token do
                mask[{1,idx}] = tonumber(token:sub(j,j))
                idx = idx + 1
                if idx > rho then
                    break
                end
            end
            dataset.mask[{insCount, {}}] = mask
        end
    end
    insCount = insCount + 1
end
local outPath = './' .. lang .. '/' .. split ..'.t7'
torch.save(outPath, dataset)
