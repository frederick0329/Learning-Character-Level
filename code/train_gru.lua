require 'nn'
require 'cutorch'
require 'rnn'
require 'optim'
require 'cunn'
local model_utils = require 'model_utils'
Train = {}

function Train.forward(clones, model, inputs, masks, targets, outputs)
    zeroTensor = torch.zeros(inputs:size()[1], rnn_size)
    if gpu then
        zeroTensor = zeroTensor:cuda()
    end
    local embeds = {}
    local hts = {}
    local cts = {}
    local input = inputs:select(2,1)
    local mask = masks:select(2,1)

    local imgs, embed
    if not use_fuse then
        imgs = clones.model.imgLookUp:forward(torch.cmul(input,(1-mask)))
        embed = clones.model.embed[1]:forward({torch.cmul(mask, input), imgs})
    else
        imgs = clones.model.imgLookUp:forward(input)
        embed = clones.model.embed[1]:forward({input, imgs})
    end
    local ht = clones.model.gru[1]:forward({embed, zeroTensor})
    table.insert(embeds, embed)
    table.insert(hts, ht)

    for step = 2,rho do 
        input = inputs:select(2,step)
        mask = masks:select(2,step)
        local imgs, embed
        if not use_fuse then
            imgs = clones.model.imgLookUp:forward(torch.cmul(input,(1-mask)))
            embed = clones.model.embed[step]:forward({torch.cmul(mask, input), imgs})
        else
            imgs = clones.model.imgLookUp:forward(input)
            embed = clones.model.embed[step]:forward({input, imgs})
        end
        ht = clones.model.gru[step]:forward({embed, hts[step-1]})
        table.insert(embeds, embed)
        table.insert(hts, ht)
    end
    local prob = model.classify:forward(hts[#hts])
    local err = model.criterion:forward(prob, targets)
    table.insert(outputs, embeds)
    table.insert(outputs, hts)
    table.insert(outputs, prob)
    return prob, err
end

function Train.backward(clones, model, inputs, masks, targets, outputs)
    zeroTensor = torch.zeros(inputs:size()[1], rnn_size)
    if gpu then
        zeroTensor = zeroTensor:cuda()
    end
    local gradInput = model.criterion:backward(outputs[3], targets)
    gradInput = model.classify:backward(outputs[2][#outputs[2]], gradInput)
    for step = rho,2,-1 do
        local input = inputs:select(2,step)
        local mask = masks:select(2,step)
        if step < rho then
            gradInput = clones.model.gru[step]:backward({outputs[1][step], outputs[2][step-1]}, gradInput[2])
        else 
            gradInput = clones.model.gru[step]:backward({outputs[1][step], outputs[2][step-1]}, gradInput)
        end
        local imgs
        if not use_fuse then
            imgs = clones.model.imgLookUp:forward(torch.cmul(input,(1-mask)))
            clones.model.embed[step]:backward({torch.cmul(mask, input), imgs}, gradInput[1])
        else
            imgs = clones.model.imgLookUp:forward(input)
            clones.model.embed[step]:backward({input, imgs}, gradInput[1])
        end
    end

    local input = inputs:select(2,1)
    local mask = masks:select(2,1)
    gradInput = clones.model.gru[1]:backward({outputs[1][1], zeroTensor}, gradInput[2])
    local imgs 
    if not use_fuse then
        imgs = clones.model.imgLookUp:forward(torch.cmul(input,(1-mask)))
        clones.model.embed[1]:backward({torch.cmul(mask, input), imgs}, gradInput[1])
    else
        imgs = clones.model.imgLookUp:forward(input)
        clones.model.embed[1]:backward({input, imgs}, gradInput[1])
    end
end

function Train.train_sgd(model, ds_train, ds_val, solver_params) 
    
    local train_size = ds_train.input:size(1)
    local val_size = ds_val.input:size(1)
    local num_batches_train = math.floor(train_size/batch_size)
    local num_batches_val = math.floor(val_size/batch_size)
    local params, grad_params = model_utils.combine_all_parameters(model.gru, model.embed, model.classify)
    print ('cloning...')
    clones = {}
    clones.model = {}
    clones.model.gru = model_utils.clone_many_times(model.gru, rho, not model.gru.parameters)
    clones.model.embed = model_utils.clone_many_times(model.embed, rho, not model.embed.parameters)
    clones.model.imgLookUp = model.imgLookUp
    print ('finish cloning...')



    for epoch = 1,num_epochs do
        local sanity_check_err = 0.0
        for n = 1,num_batches_train do
            local function feval(x)
                if x ~= params then
                    params:copy(x)
                end
                grad_params:zero()
                local inputs = ds_train.input:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, train_size-(n-1)*batch_size)):clone():double()
                local masks
                if use_img then
                    masks = torch.zeros(inputs:size())
                else
                    masks = torch.ones(inputs:size())
                end
                local targets = ds_train.target:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, train_size-(n-1)*batch_size)):clone():double()
                if gpu then 
                    inputs = inputs:cuda()
                    masks = masks:cuda()
                    targets = targets:cuda()
                end
                local outputs = {}
                local err
                _, err = Train.forward(clones, model, inputs, masks, targets, outputs)
                Train.backward(clones, model, inputs, masks, targets, outputs)
                grad_params:clamp(-5, 5)
                return err, grad_params
            end

            local _, fs = optim.adam(feval, params, solver_params)
            sanity_check_err = sanity_check_err + fs[1]
            if n % num_sanity_check == 0 then
                print(string.format("nEpoch %d; %d/%d; train err = %f ", epoch, n, num_batches_train, sanity_check_err/num_sanity_check))
                sanity_check_err = 0.0
            end
        end
        local val_err = 0
        local total_precision = 0
        for n = 1, num_batches_val do --nBatches_val do
            local inputs = ds_val.input:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, val_size-(n-1)*batch_size)):clone():double()
            local masks
            if use_img then 
                masks = ds_val.mask:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, val_size-(n-1)*batch_size)):clone():double()
            else 
                masks = torch.ones(inputs:size())
            end
            local targets = ds_val.target:narrow(1, 1+(n-1)*batch_size, math.min(batch_size, val_size-(n-1)*batch_size)):clone():double()
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
            val_err = val_err + err
            val, id = torch.max(prob, 2)
            if gpu then
                id = id:cuda()
            end
            total_precision = total_precision + torch.sum(targets:eq(id)) / math.min(batch_size, val_size-(n-1)*batch_size) 
        end
        
        print(string.format("nEpoch %d; val err = %f ", epoch, val_err/(num_batches_val)))
        print(string.format("nEpoch %d; accuracy = %f ", epoch, total_precision/(num_batches_val)))
        torch.save(model_prefix..'/model_'..epoch..'.t7', model)
    end
end

return Train
