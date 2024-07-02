import numpy as np
import os


CONFIG = {
    'play_out': 1200,        # 每次移动的模拟次数
    'dirichlet': 0.2,
    'c_puct': 5,             # u的权重
    'buffer_size': 100000,   # 经验池大小
    'model_path': r'models/current_policy.model',   # pytorch模型路径
    'train_data_buffer_path': 'train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 128,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 60,  #模型更新间隔时间
    'use_redis':  False, # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
    "theta": 1e-3,
    # "input_dim":128,
    "alpha": 1.,#强化学习 s(t) = r +  s(t+1)* alpha
    "maxstep": 20,
    "n_qubit": 2,  #量子比特的数量 2： 两个量子比特
    "matrix" : np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]])
}

