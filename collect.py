#收集数据
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
if CONFIG['use_redis']:
    import my_redis, redis

# import zip_array

class CollectPipeline:
    def __init__(self,n_qubit, theta = CONFIG['theta'], init_model=None):
        # 电路和合成环境
        self.gate_Set = env_QCS.gate_Set # 直接用qiskit的
        self.n_qubit = n_qubit
        self.theta = theta
        # self.circuit = env_QCS.circuit(self.n_qubit, self.gate_Set, self.target, self.theta)
        # self.env = env_QCS.Sys2target(self.n_qubit, self.gate_Set, self.target)  #写的太麻烦了
        # 参数
        self.temp = CONFIG['temp']  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.inital_model = init_model
        # if CONFIG['use_redis']:
        #     self.redis_cli = my_redis.get_redis_cli()

        # 从主体加载模型
    def load_model(self):
            try:
                if self.inital_model:
                    self.policy_value_net = PolicyValueNet(model_file=self.inital_model)
                    print('已加载最新模型')
                else:
                    self.policy_value_net = PolicyValueNet()
                    print('已加载初始模型')
            except:
                self.policy_value_net = PolicyValueNet()
                print('已加载初始模型', self.policy_value_net)
            ## policy_value_fn
            self.mcts_player = MCTSplayer(self.policy_value_net.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=self.n_playout,
                                          is_selfplay=1)

    def collect_selfplay_data(self, target, n_games=1):
        # 收集当前模型，当前矩阵的数据
        # n_games 处理几次当前矩阵
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            env = env_QCS.Sys2target(self.n_qubit, self.gate_Set, target)
            is_target, play_data = env.self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            with open('output.txt', 'a') as f:
                for i in range(len(play_data)):
                    print("trace:{}th\ntarget:\n{}\nstate:\n{}\nact:{}\nprobs:{}\nvalue:{}\n".format(i+1, play_data[i][4], play_data[i][0], play_data[i][1], play_data[i][2], play_data[i][3] ), file=f)
            # print("play data", play_data)
            self.episode_len = len(play_data)

            # 增加数据
            if os.path.exists(CONFIG['train_data_buffer_path']):
                    # while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                            print("success load")
                            data_file = pickle.load(data_dict)
                            self.data_buffer = deque(maxlen=self.buffer_size)
                            self.data_buffer.extend(data_file['data_buffer'])
                            # self.iters = data_file['iters']
                            del data_file
                            self.iters += 1
                            self.data_buffer.extend(play_data)
                    except FileNotFoundError:
                        print("fail load")
                        time.sleep(30)
                        # print("sleep")
            else:
                    print("initial trace set")
                    self.data_buffer.extend(play_data)
                    self.iters += 1
            # print("data buffer", self.data_buffer)
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                pickle.dump(data_dict, data_file)
                print('成功载入数据')

        return self.iters

    def run(self):
        """执行收集数据"""
        # with open(r"C:\projects\AlphaZero+QuantumCircuits\QCS_AlphaZero\dataset\intial_sequences.txt", "r") as file:
        initial_circuits = CONFIG["initial_sequences"]

        try:
            while True:
                ## 随机产生目标矩阵
                random_index = random.randint(0, len(initial_circuits) - 1)
                initial_circuit = initial_circuits[random_index]
                circuit = QuantumCircuit(1)
                for gate in initial_circuit:
                    if gate == "H":
                        circuit.h(0)
                    elif gate == "T":
                        circuit.t(0)
                    elif gate == "S":
                        circuit.s(0)
                    elif gate == "T_d":
                        circuit.tdg(0)
                    elif gate == "S_d":
                        circuit.sdg(0)
                circuit_unitary = Operator(circuit)
                target = circuit_unitary.data

                iters = self.collect_selfplay_data(target)
                # iter i：第i次添加数据，episode_len：添加了几条数据
                with open('output.txt', 'a') as f:
                    print('iter i: {}, episode_len: {}\n'.format(
                    iters, self.episode_len), file= f)

        except KeyboardInterrupt:
            print('\n\rquit')



if __name__ == '__main__':
    collecting_pipeline = CollectPipeline(CONFIG['n_qubit'], CONFIG['theta'], CONFIG['model_path'])
    collecting_pipeline.run()



