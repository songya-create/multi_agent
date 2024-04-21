from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
from common.min_buffer import min_Buffer
from common.better_buffer import better_Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.min_buffer=min_Buffer(args)
        self.better_buffer = better_Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        writer_reward = SummaryWriter("logs/draw3")
        writer_collison = SummaryWriter("logs/add_scalar4")
        writer_energy = SummaryWriter("logs/add_scalar5")
        writer_arrive = SummaryWriter("logs/add_scalar6")
        writer_shape = SummaryWriter("logs/add_scalar7")
        returns = []

        arrive_esp=0
        collis_esp =0


        for time_step in tqdm(range(self.args.time_steps)):

            # reset the environment
            if time_step % self.episode_limit == 0:
                store_better = 0
                if arrive_esp>4 and collis_esp<2:
                  self.better_buffer.store_Nepisode(self.min_buffer)
                  store_better = 1
                arrive_esp=0
                collis_esp = 0

                s = self.env.reset()
                s_next_lstm=np.zeros((self.args.seq_len, self.args.n_agents, self.args.obs_shape[0]))
                s_lstm=np.zeros((self.args.seq_len, self.args.n_agents, self.args.obs_shape[0]))
                for item in  range(self.args.seq_len):
                    s_lstm[item] = s
                for item in  range(self.args.seq_len):
                    s_next_lstm[item] = s


            #self.env.render()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    ###htack把数据按列拼接（seq_len，agent_num,obs_state）->(agent_num,obs_state*seq_len）
                    action_lstm=np.hstack(s_lstm)[agent_id]
                    action = agent.select_action(action_lstm, self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info,collis_n,energy,arrive,shape= self.env.step(actions)  #actions:[上下移动量,左右移动量,剩下3个为通信数据这里用不到]

            arrive_esp+=arrive
            collis_esp +=collis_n

            for i_lstm in range(self.args.seq_len-1):
                s_next_lstm[i_lstm]= s_next_lstm[i_lstm+1]
            s_next_lstm[self.args.seq_len-1]=s_next
            #np.hstack(s_next_lstm[0],s_next_lstm[1],s_next_lstm[2])


            self.buffer.store_episode(np.hstack(s_lstm),u,r,np.hstack(s_next_lstm))
            self.min_buffer.store_episode(np.hstack(s_lstm), u, r, np.hstack(s_next_lstm))

            """ if time_step%20==0:
            if arrive_flag==1:
            self.better_buffer.store_Nepisode(self.min_buffer)
            arrive_flag =0
            """


            #self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            s_lstm = s_next_lstm

            if self.buffer.current_size >= self.args.batch_size and time_step % 20 == 0:
                transitions = self.buffer.sample(self.args.batch_size)

                if self.better_buffer.current_size >= 2400and time_step<400000 and store_better==1:
                    transitions1=self.buffer.sample(self.args.batch_size)
                    transitions2 = self.better_buffer.sample(self.args.better_batch_size)
                    for key in transitions1.keys():
                        if 'r' in key:
                            transitions[key] = np.hstack([transitions1[key], transitions2[key]])
                        else:
                            transitions[key]=np.vstack([transitions1[key],transitions2[key]])

                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                return_1,return_2,return_3,return_4,return_5=self.evaluate()

                writer_reward.add_scalar('reward', return_1, int(time_step/self.args.max_episode_len))
                writer_collison.add_scalar('collison', return_2, int(time_step/self.args.max_episode_len))
                writer_energy.add_scalar('energy', return_3, int(time_step/self.args.max_episode_len))
                writer_arrive.add_scalar('arrive_step_count', return_4, int(time_step/self.args.max_episode_len))
                writer_shape.add_scalar('formation ', return_5, int(time_step / self.args.max_episode_len))
                """ returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')"""
            self.noise = max(0.001, self.noise - 0.000001)
            self.epsilon = max(0.001, self.epsilon - 0.000001)
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):

        returns = []
        draw_collis_count=[]
        arrive_count=0
        sum_shape=0
        episode_energy = 0
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()

            s_lstm = np.zeros((self.args.seq_len, self.args.n_agents, self.args.obs_shape[0]))
            for item in  range(self.args.seq_len):
                s_lstm[item] = s
            s_next_lstm = s_lstm
            rewards = 0
            collis_count=0
            for time_step in range(self.args.evaluate_episode_len):
               self.env.render()
               actions = []
               with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action_lstm = np.hstack(s_lstm)[agent_id]
                    action = agent.select_action(action_lstm, self.noise, self.epsilon)
                    actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, info,collis_n,energy ,arrive,shape= self.env.step(actions)
                arrive_count += arrive
                for i_lstm in range(self.args.seq_len - 1):
                    s_next_lstm[i_lstm] = s_next_lstm[i_lstm + 1]
                s_next_lstm[self.args.seq_len - 1] = s_next
                s_lstm= s_next_lstm

                rewards += sum(r) / self.args.n_agents
                collis_count += collis_n
                episode_energy+=energy
                sum_shape+=shape/3


            print(collis_count)
            draw_collis_count.append(collis_count)
            returns.append(rewards/self.args.evaluate_episode_len)
            print('Returns is', rewards/self.args.evaluate_episode_len)
            self.env.close()
        print(episode_energy/self.args.evaluate_episodes)





        return sum(returns) / self.args.evaluate_episodes,sum(draw_collis_count) / self.args.evaluate_episodes,episode_energy/self.args.evaluate_episodes,arrive_count/self.args.evaluate_episodes,sum_shape/(self.args.evaluate_episodes*self.args.evaluate_episode_len)




    def evaluate1000(self):
        self.noise = 0.001
        self.epsilon = 0.001
        test_count=1000
        returns = []
        draw_collis_count=[]
        finish_target=[]

        sum_shape=0
        episode_energy = 0
        for episode in range(test_count):
            # reset the environment
            s = self.env.reset()
            s_lstm = np.zeros((self.args.seq_len, self.args.n_agents, self.args.obs_shape[0]))
            for item in  range(self.args.seq_len):
                s_lstm[item] = s
            s_next_lstm = s_lstm
            rewards = 0
            collis_count=0
            arrive_count = 0
            finish_target_flag = 0
            for time_step in range(self.args.evaluate_episode_len):
               #self.env.render()
               actions = []
               with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action_lstm = np.hstack(s_lstm)[agent_id]
                    action = agent.select_action(action_lstm, self.noise, self.epsilon)
                    actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, info,collis_n,energy ,arrive,shape= self.env.step(actions)
                arrive_count += arrive
                for i_lstm in range(self.args.seq_len - 1):
                    s_next_lstm[i_lstm] = s_next_lstm[i_lstm + 1]
                s_next_lstm[self.args.seq_len - 1] = s_next
                s_lstm= s_next_lstm

                rewards += sum(r) / self.args.n_agents
                collis_count += collis_n
                episode_energy+=energy
                sum_shape+=shape/3
                if collis_count<2 and arrive_count>4 and finish_target_flag==0:
                    finish_target.append(time_step)
                    finish_target_flag=1


            draw_collis_count.append(collis_count)
            returns.append(rewards/self.args.evaluate_episode_len)
            self.env.close()
            print('episode    ---------', episode)
            print('collis_count', collis_count)
            print('arrive_count', arrive_count)


        print('ending\n')
        print('finish target count',len(finish_target))
        print('average formation error',sum_shape/(test_count*self.args.evaluate_episode_len))
        print('average step in finish target',sum(finish_target)/len(finish_target))
        print('energy',episode_energy/test_count)

        return None
