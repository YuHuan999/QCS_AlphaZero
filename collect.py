#收集自我对弈数据
import random
import numpy as np
from collections import deque
import copy
import os
import pickle
import time
from network import PolicyValueNet
import env_QCS
from MCTS import MCTSplayer
from config import CONFIG

if CONFIG['use_redis']:
    import my_redis, redis

# import zip_array

class CollectPipeline:
    def __init__(self,n_qubit, target, theta = CONFIG['theta'], init_model=None):
        # 电路和合成环境
        self.gate_Set = env_QCS.gate_Set
        self.n_qubit = n_qubit
        self.target = target
        self.theta = theta
        # self.circuit = env_QCS.circuit(self.n_qubit, self.gate_Set, self.target, self.theta)
        self.env = env_QCS.Sys2target(self.n_qubit, self.gate_Set, self.target)
        # 参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()

        # 从主体加载模型
    def load_model(self, model_file=CONFIG["model_path"]):
            try:
                self.policy_value_net = PolicyValueNet(model_file)
                print('已加载最新模型')
            except:
                self.policy_value_net = PolicyValueNet()
                print('已加载初始模型', self.policy_value_net)
                # print(self.policy_value_net)
            self.mcts_player = MCTSplayer(self.policy_value_net.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=self.n_playout,
                                          is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        print("buffer size" ,self.buffer_size)

        # 收集自我对弈的数据
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            is_target, play_data = self.env.self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # 增加数据
            if os.path.exists(CONFIG['train_data_buffer_path']):
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = deque(maxlen=self.buffer_size)
                                self.data_buffer.extend(data_file['data_buffer'])
                                self.iters = data_file['iters']
                                del data_file
                                self.iters += 1
                                self.data_buffer.extend(play_data)
                            print('成功载入数据')
                            break
                        except:
                            time.sleep(30)
            else:
                    self.data_buffer.extend(play_data)
                    self.iters += 1
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                pickle.dump(data_dict, data_file)

        return self.iters

    def run(self):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(
                    iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')



if __name__ == '__main__':
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])


    collecting_pipeline = CollectPipeline(2, matrix, init_model='current_policy.model')
    collecting_pipeline.run()



