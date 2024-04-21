
##description
3个蓝色智能体，避开黑色障碍物，在保持三角形编队的前提下，追踪围绕红色智能体。在完成任务期间不发生碰撞
##Requirements
- python=3.7
- torch=1.12.0+cu116
# MADDPG
This is a pytorch implementation of MADDPG on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of MADDPG is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).
基于环境修改制定围剿任务。

围剿任务代码：simple_task1.py,将其放入MPE配置环境**\multiagent-particle-envs-master\multiagent\scenarios文件夹下
            
            在本项目中将common.arguments.py中修改
            
             parser.add_argument("--scenario-name", type=str, default="simple_task1.py",）
             
##在MADDPG上加入了lstm与双经验池

##train

将common.arguments.py中

parser.add_argument("--evaluate", type=bool, default=False)

具体的训练参数也在这个common.arguments.py文件中修改

##evaluate

对模型的评估

parser.add_argument("--evaluate", type=bool, default=True

## Quick Start

执行main.py

Directly run the main.py, then the algrithm will be tested on scenario 'simple_tag' for 10 episodes, using the pretrained model.

## tensorbord查看训练记录

在terminal窗口输入：tensorboard --logdir [log路径] 
![image](https://github.com/songya-create/multi_agent/assets/63812791/340e220e-dccc-4587-8281-0f361f78bdc7)


