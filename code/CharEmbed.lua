require 'nngraph'
require 'nn'

local CharEmbed = {}

function CharEmbed.embed(nIndex, rnn_size)
        local inputs = {}
        table.insert(inputs, nn.Identity()())   -- network input
        local char_inputs = inputs[1]
        local char_embed = nn.LookupTableMaskZero(nIndex, rnn_size)(char_inputs)
        outputs = {}
        table.insert(outputs, char_embed)
        -- packs the graph into a convenient module with standard API (:forward(), :backward())
        return nn.gModule(inputs, outputs)
end

return CharEmbed
