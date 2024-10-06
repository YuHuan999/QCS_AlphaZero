import numpy as np
import os


CONFIG = {
    'play_out': 10,        # 每次移动的模拟次数
    'dirichlet': 0.2,
    'c_puct': 5,   # u的权重

    'temp': 1e-3,
    'buffer_size': 100000,   # 经验池大小
    'batch_size': 1024,

    'model_path': 'models/current_policy.model',   # 最新模型地址
    'train_data_buffer_path': 'train_data_buffer.pkl',   # 训练数据地址

    "learn_rate": 1e-3,
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 对于相同的神经网输入训练的次数
    "check_freq": 10, # 保存模型的频率
    # 'game_batch_num': 3000,  # 训练更新的次数


    'train_update_interval': 60,  #模型更新间隔时间

    "theta": 0.0001, # 判别生成矩阵是否达到target
    "discount_rl": 1.0,
    "alpha": 0.85,# alphazero中q值传播率，子向父传播
    "maxstep": 5,
    "n_qubit": 1,  #量子比特的数量
    "input_dim": 8, #(n_qubit*2)^2 * 2
    "output_dim": 5, #(n_qubit*5) + (n_qubit - 1) * n_qubit

    "device": "cuda",



    "gateset":["H", "S", "S_d", "T", "T_d"],
    "gateset_env": [["H", 0], ["S", 0], ["S_d", 0], ["T", 0], ["T_d", 0]],
    "gateset_matrix":[ np.array([[0.70710678+0.j, 0.70710678+0.j], [0.70710678+0.j, -0.70710678+0.j]]), #H
                       np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.+1.j]]),                  #S
                       np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.-1.j]]),                 #S_d
                       np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.70710678+0.70710678j]]),  #T
                       np.array([[1.+0.j, 0.+0.j], [0.+0.j, 0.70710678-0.70710678j]])  #T_d
],

    "matrix" : np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]]),
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'use_redis': False,  # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,

    "initial_sequences": [
                                ['T_d', 'T_d', 'S_d', 'H', 'S_d'],
                                ['T_d', 'H'],
['S_d', 'T'],
['T_d', 'T_d', 'T_d', 'T'],
['S', 'S_d', 'H', 'T', 'H'],
['S', 'H'],
['H', 'T_d'],
['S', 'S', 'S'],
['S', 'S', 'H'],
['T', 'S', 'T'],
['T', 'S_d', 'S_d'],
['S', 'S_d'],
['T', 'T_d', 'T', 'S_d', 'S_d'],
['S_d', 'S', 'T', 'H'],
['T_d', 'H'],
['S', 'S_d', 'S_d', 'T_d', 'T'],
['T', 'T', 'T', 'H'],
['S_d', 'S_d', 'T_d', 'S'],
['S_d', 'S_d', 'T_d'],
['S_d', 'S_d'],
['S_d', 'S'],
['T', 'S', 'H', 'H'],
['T', 'S'],
['S_d', 'T_d', 'T', 'T'],
['T', 'S', 'T', 'S'],
['S_d', 'S_d', 'S', 'T_d', 'T_d'],
['S', 'T_d', 'T_d', 'S'],
['H', 'S_d'],
['T', 'H', 'T_d', 'H'],
['S', 'T', 'S', 'T_d', 'T_d'],
['S', 'H', 'T', 'T'],
['S_d', 'T_d', 'S', 'T'],
['S', 'T_d', 'T', 'S_d', 'S'],
['S_d', 'H', 'T_d'],
['S', 'T_d', 'T', 'T', 'T'],
['T', 'T_d', 'H', 'S_d'],
['S', 'H', 'S', 'S_d', 'H'],
['H', 'S', 'T', 'T'],
['S_d', 'T_d', 'H', 'S', 'S_d'],
['T', 'S', 'S', 'T_d'],
['T_d', 'H'],
['T_d', 'S_d', 'T', 'S_d', 'H'],
['S', 'T'],
['S_d', 'T_d', 'S', 'S', 'S'],
['S_d', 'S_d', 'S_d', 'H', 'S_d'],
['S', 'T_d', 'T_d', 'H', 'T'],
['S_d', 'H', 'T', 'T', 'S_d'],
['T', 'S_d'],
['S_d', 'S_d', 'H'],
['T_d', 'S', 'S_d'],
    ]
}

