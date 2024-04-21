import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.lstm = nn.GRU(input_size = args.obs_shape[agent_id], hidden_size = 64, num_layers = 1,batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        #self.fc2 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        ##输入的是x=（batch，seq*obs_size）->x=(batch,time_step,input_size)  ####批量存储的数据处理
        x = x.view(-1,self.args.seq_len, self.args.obs_shape[0])
        out,_=self.lstm(x)###out=(batch_size,seq_len,hidden_size)
        out=out[:,-1,:]####(batch_size,hidden_size)

        x = F.relu(self.fc1(out))
        #x = F.relu(self.fc2(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args=args
        self.max_action = args.high_action
        #输入信息维度，输出的隐藏状态维度，层数（时序信息会根据输入自动调整长度n）输入格式（batch,seq,input_size),
        # batch_first = True含义是输入的信息格式为（batch，seq，input_size）
        self.lstm = nn.GRU(input_size = sum(args.obs_shape), hidden_size = 64, num_layers = 1, batch_first = True)
        self.fc1 = nn.Linear(sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64+64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):


        #for agent_id in range(self.args.n_agents):
        state_lstm0 = state[0].view(-1, self.args.seq_len, self.args.obs_shape[0])
        state_lstm1 = state[1].view(-1, self.args.seq_len, self.args.obs_shape[1])
        state_lstm2 = state[2].view(-1, self.args.seq_len, self.args.obs_shape[2])

        state = torch.cat([state_lstm0,state_lstm1,state_lstm2], dim=2) #希望得到（batch,seq,input_size),3维数据怎么cat？

        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        out1,_=self.lstm(state)
        out1 = out1[:, -1, :]
        out2=self.fc1(action)
        x = torch.cat([out1, out2], dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
