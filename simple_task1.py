import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math
import cv2
import numpy as np
import  random

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 4)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY,vx,vy):
        '''This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)],[np.float32(vx)],[np.float32(vy)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        #x, y = predicted[0], predicted[1]
        return predicted[0], predicted[1],predicted[2], predicted[3]


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3

        world.num_agents = 3
        num_landmarks = 1
        # 修改二增加障碍物
        num_obstruction = 3 #障碍物设置
        world.collaborative = True



        # add agents
        world.agents = [Agent() for i in range(num_agents)] #创建智能体
        for i, agent in enumerate(world.agents):
            agent.u_noise = 0
            agent.max_speed=0.1
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
            agent.num=i

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = True
            landmark.size = 0.02
            landmark.num = i
        world.obstruction = [Landmark() for i in range(num_obstruction)]
        for i, obstruction in enumerate(world.obstruction):
            obstruction.name = 'obstruction %d' % i
            obstruction.collide = False
            obstruction.movable = True
            obstruction.size  = 0.02 #
        num_obspre=3
        world.obs_pre= [Landmark() for i in range(num_obspre)]

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

            agent.posx = np.zeros(400)
            agent.posy =  np.zeros(400)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.8, 0.25, 0.25])
        for i, obstruction in enumerate(world.obstruction):
            obstruction.color = np.array([0.1,0.1,0.1])
        # set random initial states
        for i, agent in enumerate(world.agents):
            #修改一固定初始位置
            agent.state.p_pos=np.array([0.9-0.3*i,-0.88])
            #agent.state.p_pos = np.random.uniform(-0.9, -0.1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        B = random.sample(range(1, 8), 3)
        world.landmarks[0].state.p_pos =np.array([0.7, 0.65])#静态np.array([0.0, 0.0])# 动态world.landmarks[0].state.p_pos =np.array([0.7, 0.7])
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
       # world.landmarks[1].state.p_pos = np.array([0.8, 0.729])
       # world.landmarks[2].state.p_pos =np.array([0.9,0.9])
        #for i, landmark in enumerate(world.landmarks):
            # 修改二固定目标位置
            #landmark.state.p_pos = np.random.uniform(0.5, 0.8, world.dim_p)
            #landmark.state.p_pos = np.array([-0.8+0.2*B[i],0.8])
            #landmark.state.p_pos = np.array([0.5 + 0.2 *i, 0.8])
            #landmark.state.p_vel = np.zeros(world.dim_p)
        obsv=np.random.uniform(0.02, 0.1, 3)
        for i, obstruction in enumerate(world.obstruction):
           # obstruction.state.p_pos =np.array([0-0.9*(-1)**(i),0.3-0.4*i]) ##np.random.uniform(-0.8,+0.8, world.dim_p)
           obstruction.state.p_pos =np.random.uniform(-0.8,+0.4, world.dim_p)#np.array([0-0.9*(-1)**(i),0.35-0.45*i])##静态np.array([0 - 0.6 * (-1) ** (i), 0.5- 0.5 * i])###动态obstruction.state.p_pos =np.array([0-0.9*(-1)**(i),0.3-0.4*i])
           obstruction.state.p_vel = np.zeros(world.dim_p)
           obstruction.vbase=obsv[i]
        self.time_t = 0

        for i, agent in enumerate(world.agents):
            agent.v_last=[0,0]
            agent.E_rew = [0, 0]
            agent.E_shape_rew = [0, 0]
            ####一些奖励函数参数设置
            if agent.name == 'agent 0':
                state_pos0 = world.landmarks[0].state.p_pos + (0.3 / 2, -np.sqrt(3) * 0.3 / 6)
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - state_pos0)))
            if agent.name == 'agent 1':
                state_pos1 = world.landmarks[0].state.p_pos + (0, np.sqrt(3) * 0.3 / 3)
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - state_pos1)))
            if agent.name == 'agent 2':
                state_pos2 = world.landmarks[0].state.p_pos + (-0.3 / 2, -np.sqrt(3) * 0.3 / 6)
                dists = np.sqrt(np.sum(np.square(agent.state.p_pos - state_pos2)))
            abs_dis = np.abs(dists)
            # agent.E_rew[1] = np.exp(-(abs_dis)) - (abs_dis)  # 构建势能场
            agent.E_rew[0] = -np.square(5 * abs_dis) - (abs_dis)

            for other in world.agents:
                if other is agent: continue
                # comm.append(other.state.c)
                agent.E_shape_rew[0] +=-np.abs(np.sum(np.sqrt(np.square(agent.state.p_pos - other.state.p_pos)))-0.3)



    # 计算reward以及相撞
    def benchmark_data(self, agent, world):
        rew= 0
        collisions = 0
        collisions_obs=0
        occupied_landmarks = 0
        min_dists = 0
        done=False

        ###能耗惩罚
        #rew-=0.01*(abs(agent.state.p_force[0])+abs(agent.state.p_force[1]))
        '''
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= 0.5*min(dists)
        '''

        '''
        for i, l in enumerate(world.landmarks):
            if l.num==agent.num:
               dists =np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
               #rew= max(1.4-dists,0)
               rew-=dists
               if dists < 0.1:
                occupied_landmarks += 1
        '''
        ####围绕目标点
        if agent.name=='agent 0':
          state_pos0=world.landmarks[0].state.p_pos + (0.3/2, -np.sqrt(3) * 0.3 / 6)
          dists = np.sqrt(np.sum(np.square(agent.state.p_pos -state_pos0 )))
        if agent.name=='agent 1':
          state_pos1 = world.landmarks[0].state.p_pos+(0,np.sqrt(3) * 0.3 / 3)
          dists = np.sqrt(np.sum(np.square(agent.state.p_pos -state_pos1 )))
        if agent.name=='agent 2':
           state_pos2 =world.landmarks[0].state.p_pos + (-0.3/2, -np.sqrt(3) * 0.3 / 6)
           dists = np.sqrt(np.sum(np.square(agent.state.p_pos -state_pos2)))

        abs_dis = np.abs(dists)
        if abs_dis == 0:
            rew+=1
        #agent.E_rew[1] = np.exp(-(abs_dis)) - (abs_dis)  # 构建势能场
        agent.E_rew[1]=-np.square(5*abs_dis)- (abs_dis)

        # rew= max(1.4-dists,0)
        rew +=(agent.E_rew[1]-agent.E_rew[0])
        agent.E_rew[0] = agent.E_rew[1]
        rew += np.exp(-(4*abs_dis))
        agent.E_shape_rew[1] = 0

        ####三角形形状
        for other in world.agents:
            if other != agent:
             agent.E_shape_rew[1] +=-np.abs(np.sum(np.sqrt(np.square(agent.state.p_pos - other.state.p_pos))) - 0.3)

        rew +=10*(agent.E_shape_rew[1] - agent.E_shape_rew[0])
        agent.E_shape_rew[0]=agent.E_shape_rew[1]

        san_shape= -agent.E_shape_rew[1]
        #rew-=0.1*san_shape
        

        if agent.collide:
            """for i,a in  enumerate(world.agents):
                if not(a.num ==agent.num):
                    rew -= self.is_collision(a, agent)[1]
                    if self.is_collision(a, agent)[0]:
                        collisions += 1"""
            if self.is_coll_wall(agent):
                rew -= 1
                collisions += 1
                
            for i, obstruction in enumerate(world.obstruction):
                coll_obs= self.is_collision(obstruction, agent)
                rew -=coll_obs[1]
                if coll_obs[0]:
                    collisions_obs += 1
                    #rew -= 1
                    
        #energy = (abs((agent.state.p_vel[0]) * agent.state.p_force[0]) + abs(
        #    (agent.state.p_vel[1]) * agent.state.p_force[1])) * 10
        #x轴上
        if agent.v_last[0]*agent.state.p_vel[0]>0:
            energy=abs(np.square(agent.state.p_vel[0]*10)-np.square(agent.v_last[0]*10))
        else:
            energy = abs(np.square(agent.state.p_vel[0] * 10) + np.square(agent.v_last[0] * 10))
        # y轴上
        if agent.v_last[1]*agent.state.p_vel[1]>0:
            energy+=abs(np.square(agent.state.p_vel[1]*10)-np.square(agent.v_last[1]*10))
        else:
            energy+= abs(np.square(agent.state.p_vel[1] * 10) + np.square(agent.v_last[1] * 10))
        energy1=5*0.5*abs(np.sum(np.square(agent.state.p_vel * 10))- np.sum(np.square(agent.v_last * 10)))
        energy=5*0.5* energy
        agent.v_last=agent.state.p_vel
         ###实际模仿场景为20*20m场景
        rew-=energy1*0.00001
        return (rew, collisions_obs,occupied_landmarks,energy1,abs_dis,san_shape)#, min_dists, occupied_landmarks)
    
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        if dist < dist_min:
            return [True,np.exp(-10*dist)]
        if  dist_min<=dist and dist < dist_min:###动态这里是dist <  3*dist_min
            return [False,np.exp(-10*dist)]
        else:
            return  [False,0]
    #撞击墙壁惩罚
    def is_coll_wall(self,a):
        if  all(abs(i) < 1 for i in a.state.p_pos):
            return False
        else:
            return  True

    def is_coll_pre(self, a, world):
        for i, obstruction in enumerate(world.obs_pre):
            if (self.is_collision(obstruction, a)):
                return True
        return False



    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew


    def updat_obstruction(self,world):
        self.time_t+=1
        pre0=[]
        pre1=[]
        pre2=[]
        for i, obstruction in enumerate(world.obstruction):
            noise = 0.5*np.random.randn(2)  # gaussian noise
            obstruction.state.p_pos+=world.dt*obstruction.state.p_vel##np.random.uniform(-0.8,+0.8, world.dim_p)
            Vx=(obstruction.vbase)*(1)*((-1)**i)*((-1)**(int(self.time_t/150)))
            #Vy=0.02*math.cos(0.03*self.time_t)
            Vy=0.003*(1+noise[1])
            #Vy=0.02*((-1)**i)*((-1)**(int(self.time_t/200)))
            obstruction.state.p_vel= np.array([Vx,Vy])
            
        world.landmarks[0].state.p_pos += world.dt*world.landmarks[0].state.p_vel
        noise =0.5*np.random.randn(2)
        Vx = -0.015*(1) * ((-1) ** i) * ((-1) ** (int(self.time_t / 1000)))
        # Vy=0.02*math.cos(0.03*self.time_t)
        #Vy = 0.005*(1+noise[1])*math.cos(0.01*self.time_t)
        #Vy = 0.015 * (noise[1])
        world.landmarks[0].state.p_vel = np.array([Vx,Vy])
           
    def observation(self, agent, world):
       # self.get_path(agent,times = self.time_t)
        #if self.time_t==399:
        #    print("----this is agent",agent.name)
         #   print(agent.posx)
          #  print(agent.posy)

        dark_dis=[100,100]
        if agent.name=='agent 1':
         self.updat_obstruction(world)
        # get positions of all entities in this agent's reference frame
        entity_pos = []
         ##只用管自己的目标
        for entity in world.landmarks:  # world.entities:
           # if self.visionable_area(world,entity):
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            #else:
            #    entity_pos.append(dark_dis)


        entity_pos2 = []

        for entity in world.obstruction :  # world.entities:
            if self.visionable_area(world,entity):
                entity_pos2.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos2.append(dark_dis)




        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:

            if other is agent: continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + entity_pos2 + other_pos + comm)


    def visionable_area(self,world, a_vis):
            for agent in world.agents:
                if np.sqrt(np.sum(np.square(agent.state.p_pos-a_vis.state.p_pos))) <2:
                  return True
            return False

    def get_path(self, agent,times):
        agent.posx[times] = agent.state.p_pos[0]
        agent.posy[times] = agent.state.p_pos[1]
        return agent.posx,agent.posy