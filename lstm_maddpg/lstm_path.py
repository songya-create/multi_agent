import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class lstm_path(nn.Module):
    def __init__(self, args, agent_id):
        super(lstm_path, self).__init__()

        self.lstm = nn.GRU(input_size = 2*4, hidden_size = 64, num_layers = 1,batch_first=True)

        self.path_out = nn.Linear(64, 3*2)###未来3步预测

    def forward(self, x):
        ##输入的是x=（batch，seq*obs_size）->x=(batch,time_step,input_size)  ####批量存储的数据处理
        x = x.view(-1,self.args.seq_len, self.args.obs_shape[0])
        out,_=self.lstm(x)###out=(batch_size,seq_len,hidden_size)
        out=out[:,-1,:]####(batch_size,hidden_size)

        out_data = self.path_out(out)

        return out_data
