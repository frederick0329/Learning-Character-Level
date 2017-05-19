--require 'json'
require 'image'

--path = 'utf2idx.json'
--dict = json.load(path)
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-n', 100)
cmd:option('-l', 'zh_traditional')
opt = cmd:parse(arg)
num_chars = opt.n
lang = opt.l
--num_chars = 8986
--lang = 'zh_traditional'


size = 36
imgs = torch.Tensor(num_chars, 3, size, size)

for i=1,num_chars do
    img = image.load(lang .. '/' .. '/img/' ..i.. '.jpg') 
    imgs[{{i},{},{},{}}] = img
end

torch.save(lang .. '/' .. '/img_'..size..'.t7', imgs)
