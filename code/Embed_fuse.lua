require 'nngraph'
require 'nn'
require 'rnn'

local ImgEmbed
if img_size == 72 then
    ImgEmbed = require 'ImgEmbed'
else
    ImgEmbed = require 'ImgEmbed_36'
end
local CharEmbed = require 'CharEmbed'

local Embed = {}

function Embed.embed(nIndex, embed_size)
        local inputs = {}
        table.insert(inputs, nn.Identity()())   
        table.insert(inputs, nn.Identity()()) 
        local char_inputs = inputs[1]
        local img_inputs = inputs[2]
        local char_embed = char_pretrained_model
        local img_embed
        if use_cnn then
            img_embed = cnn_pretrained_model
        else
            img_embed = nn.MaskZero(nn.Linear(3*img_size*img_size, embed_size), 1)(img_inputs)
        end
        
        non_linear_img_embed = nn.Tanh()(img_embed)
        non_linear_char_embed = nn.Tanh(char_embed)
        local joint_embed = nn.JoinTable(2)({non_linear_char_embed, non_linear_img_embed})
        local out_embed = nn.Linear(2*embed_size, embed_size)(joint_embed)
        local non_linear_out_embed = nn.Tanh()(out_embed)
        outputs = {}
        table.insert(outputs, non_linear_out_embed)
        return nn.gModule(inputs, outputs)
end

return Embed
