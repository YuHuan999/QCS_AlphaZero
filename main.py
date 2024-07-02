# main.py
# 运行训练好的模型
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

class CircuitSys:
    def __init__(self,n_qubit, target, theta = CONFIG['theta']):
        # 电路和合成环境
        self.gate_Set = env_QCS.gate_Set
        self.n_qubit = n_qubit
        self.target = target
        self.theta = theta
        self.circuit = env_QCS.circuit(self.n_qubit, self.gate_Set, self.target, self.theta)
        self.env = env_QCS.Sys2target(self.n_qubit, self.gate_Set, self.target)

        #超参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG['play_out']  # 每次移动的模拟次数
        self.c_puct = CONFIG['c_puct']  # u的权重
        self.buffer_size = CONFIG['buffer_size']  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
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
    def synthesize(self, temp = 1e-3):
        is_target = False
        # circuit = self.circuit
        self.env.circuit.init_circuit()
        while not is_target:
            actions = []
            moves = self.mcts_player.get_action(self.env.circuit, self.env.num_action, temp = temp, return_prob = 1)
            action = self.env.circuit.dict_id2actions.get(moves[0])
            print("action", action)
            actions.append(action)
            self.env.circuit.insert_gate(action[0], action[1])
            res = self.env.circuit.is_target()
            is_target = res[0]
            print("is_target", is_target)
            print(self.env.circuit.state)

        return actions







if __name__ == "__main__":
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    test1 = CircuitSys(2, matrix)
    test1.load_model()
    # test1.load_model(r'models/current_policy2.model')
    actions = test1.synthesize()

    # print(actions)











