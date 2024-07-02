#使用收集到的数据训练
import torch
import random
from collections import defaultdict, deque
import numpy as np
import pickle
import time
from network import prepare_input

import zip_array
from config import CONFIG
import env_QCS
from MCTS import MCTSplayer
from network import PolicyValueNet

def prepare_inputs(matrics):
    inputs = []
    for matrix in matrics:


        real_part = matrix.real
        imag_part = matrix.imag
        flattened_real = real_part.flatten()
    # print("f",flattened_real)
        flattened_imag = imag_part.flatten()
    # print("f",flattened_imag)
        input = torch.cat((torch.tensor(flattened_real, dtype=torch.float32), torch.tensor(flattened_imag, dtype=torch.float32)), dim=-1)
        inputs.append(input)
    return inputs

def get_observations(states, target):
    # print("state", circuit.state)
    # print("target", circuit.target)
    observations = []
    for state in states:
        observation = np.dot(state, target)
        observations.append(observation)
    return observations


class TrainPipeline:

    def __init__(self, n_qubit, target, theta = CONFIG['theta'], model_path = None):
        # 电路和合成环境
        self.gate_Set = env_QCS.gate_Set
        self.n_qubit = n_qubit
        self.target = target
        self.theta = theta
        # self.circuit = env_QCS.circuit(self.n_qubit, self.gate_Set, self.target)
        self.env = env_QCS.Sys2target(self.n_qubit, self.gate_Set, self.target)
        # 参数
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']  # 训练的batch大小
        self.epochs = CONFIG['epochs']  # 每次更新的train_step数量
        self.kl_targ = CONFIG['kl_targ']  # kl散度控制
        self.check_freq = 100  # 保存模型的频率
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数

        self.buffer_size = maxlen=CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        # model_path = r'models/current_policy.model'

        if model_path:
            print("model_path",model_path)
            try:
                self.policy_value_net = PolicyValueNet('models\\current_policy.model')
                print('已加载上次最终模型',model_path)
            except:
                # 从零开始训练
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()


    def network_updata(self):
        """更新策略价值网络"""
        print("网络更新")
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        print("mini_batch, type", type(mini_batch))
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]

        state_batch = [data[0] for data in mini_batch]
        print("before pre state",type(state_batch), len(state_batch), state_batch, )
        state_batch = get_observations(state_batch, CONFIG["matrix"])
        print("after get observations", type(state_batch), len(state_batch), state_batch, )
        state_batch = prepare_inputs(state_batch)
        print("after prepare", type(state_batch), len(state_batch), state_batch, )
        state_batch = torch.stack(state_batch, dim=0)

        # state_batch = np.ascontiguousarray(state_batch)
        # state_batch = torch.as_tensor(state_batch).to(self.device)
        # state_batch = prepare_input(get_observation(state_batch, CONFIG["matrix"]))
        # print("after pre state", state_batch, len(state_batch))
        # print("state_batch, type, before", state_batch, type(state_batch))

        # print("state_batch, type", state_batch, type(state_batch))

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        value_batch = [data[2] for data in mini_batch]
        value_batch = np.array(value_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                value_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(value_batch) - old_v.flatten()) /
                             np.var(np.array(value_batch)))
        explained_var_new = (1 -
                             np.var(np.array(value_batch) - new_v.flatten()) /
                             np.var(np.array(value_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.9f},"
               "explained_var_new:{:.9f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        """开始训练"""
        print("train.........................")
        try:
            for i in range(self.game_batch_num):
                while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = data_file['data_buffer']
                                print("buffer_size", self.buffer_size)
                                print("size: data_buffer", len(self.data_buffer))
                                print("size: batch_size", self.batch_size)
                                self.iters = data_file['iters']
                                del data_file
                            print('已载入数据')
                            break
                        except:
                            time.sleep(30)


                print('step i {}: '.format(self.iters))

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.network_updata()

                time.sleep(CONFIG['train_update_interval'])  # 每1分钟更新一次模型

                if (i + 1) % self.check_freq == 0:
                    print("save model")
                    print("current self-play batch: {}".format(i + 1))
                    self.policy_value_net.save_model('models/current_policy_batch{}.model'.format(i + 1))
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    training_pipeline = TrainPipeline(2, matrix, CONFIG['theta'], CONFIG["model_path"])
    training_pipeline.run()