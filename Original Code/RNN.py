
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args = args
        input_size = args.enc_in
        output_size = args.c_out
        hidden_size = args.hidden_size
        n_layers = args.n_layers
        self.lstm = nn.RNN(input_size,hidden_size,n_layers,batch_first=True)
        self.reg = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x,_ = self.lstm(x)
        b,s,h = x.shape
        x = self.reg(x)
        return x[:,-self.args.pred_len:,:]
