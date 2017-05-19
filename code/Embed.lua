require 'nngraph'
require 'nn'
require 'rnn'

local ImgEmbed = require 'ImgEmbed_36_deep'
local CharEmbed = require 'CharEmbed'

local Embed = {}

function Embed.embed(nIndex, embed_size)
        local inputs = {}
        table.insert(inputs, nn.Identity()())   
        table.insert(inputs, nn.Identity()()) 
        local char_inputs = inputs[1]
        local img_inputs = inputs[2]
        local char_embed = CharEmbed.embed(nIndex, embed_size)(char_inputs)
        local img_embed = nn.MaskZero(ImgEmbed.embed(embed_size), 1)(img_inputs)
        
        non_linear_img_embed = nn.Tanh()(img_embed)
        local joint_embed = nn.CAddTable()({char_embed, non_linear_img_embed})
        outputs = {}
        table.insert(outputs, joint_embed)
        return nn.gModule(inputs, outputs)
end

return Embed
