#使用收集到的数据训练
import torch
import random
from collections import defaultdict, deque
import numpy as np
import pickle
import time
import cirq
from network import prepare_input

import zip_array
from config import CONFIG
import env_QCS
from MCTS import MCTSplayer
from network import PolicyValueNet
from torch.utils.tensorboard import SummaryWriter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
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

def get_observations(states, targets):
    # print("state", circuit.state)
    # print("target", circuit.target)
    observations = []
    for state in states:
        observation = np.dot(state, targets)
        observations.append(observation)
    return observations

def recovery_state_mcts_prob(tuple):

    state, move, mcts_prob, values = tuple

    return state, mcts_prob, values

class TrainPipeline:

    def __init__(self, model_path = None):
        # 电路和合成环境
        # self.gate_Set = env_QCS.gate_Set
        # self.n_qubit = n_qubit
        # self.target = target
        # self.theta = theta
        # self.circuit = env_QCS.circuit(self.n_qubit, self.gate_Set, self.target)
        # self.env = env_QCS.Sys2target(self.n_qubit, self.gate_Set, self.target)
        # 参数
        # self.n_playout = CONFIG['play_out']
        # self.c_puct = CONFIG['c_puct']
        self.learn_rate = CONFIG['learn_rate']
        self.temp = CONFIG['temp']
        self.batch_size = CONFIG['batch_size']  # 训练的batch大小
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率 ??



        self.epochs = CONFIG['epochs']  # 对于相同的神经网输入训练的次数
        self.kl_targ = CONFIG['kl_targ']  # kl散度控制
        self.check_freq = CONFIG['check_freq']  # 保存模型的频率
        # self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数???

        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iter = 0

        if model_path:
            # print("model_path",model_path)
            try:
                self.policy_value_net = PolicyValueNet(model_path)
                print('已加载上次最终模型',model_path)
            except:
                # 从零开始训练
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()


    def network_updata(self):  ## data structure of input for network exists bugs
        """更新策略价值网络"""
        # print("网络更新")
        # 从data_buffer取出batch
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [recovery_state_mcts_prob(data) for data in mini_batch]
        # 简洁化测试的时候试一试
        # mini_batch = [(state, mcts_prob, values) for state, move, mcts_prob, values in mini_batch]


        #准备神经网的输入
        target_batch = mini_batch[:,4]  # U
        state_batch = mini_batch[:, 0]  # V
        # print("before pre state",type(state_batch), len(state_batch), state_batch, )
        ## bug?
        #
        # state_batch = np.array([s.conj().T for s in state_batch])
        conj_matrices = np.conjugate(state_batch) #共轭每一个矩阵
        # 转置每个矩阵
        state_batch = conj_matrices.transpose(0, 2, 1) #V†
        state_batch = np.sum(state_batch * target_batch , axis=1) #V†U
        # state_batch = get_observations(state_batch, target_batch)
        real_part = state_batch.real  # 提取实部
        imag_part = state_batch.imag  # 提取虚部
        # 展平实部和虚部
        flattened_real = real_part.view(real_part.shape[0], -1)
        flattened_imag = imag_part.view(imag_part.shape[0], -1)
        # 连接实部和虚部
        state_batch = torch.cat((flattened_real, flattened_imag), dim=-1)  #flatten
        # state_batch = prepare_inputs(state_batch)
        state_batch = torch.stack(state_batch, dim=0)



        # print("after get observations", type(state_batch), len(state_batch), state_batch, )

        # print("after prepare", type(state_batch), len(state_batch), state_batch, )


        # state_batch = np.ascontiguousarray(state_batch)
        # state_batch = torch.as_tensor(state_batch).to(self.device)
        # state_batch = prepare_input(get_observation(state_batch, CONFIG["matrix"]))
        # print("after pre state", state_batch, len(state_batch))
        # print("state_batch, type, before", state_batch, type(state_batch))

        # print("state_batch, type", state_batch, type(state_batch))

        mcts_probs_batch = mini_batch[:, 1]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        value_batch = mini_batch[:, 2]
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


        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,))

        return loss, entropy

    def run(self):
        print(".............train......start...................")
        ## 准备测试集
        #  test matrix
        testset = np.load(
            r"C:\projects\AlphaZero+QuantumCircuits\QCS_AlphaZero\dataset\target_matrices_qis.npy")
        #  测试集中矩阵的原始电路的量子门数
        with open(r"C:\projects\AlphaZero+QuantumCircuits\QCS_AlphaZero\dataset\intial_sequences.txt",
                  "r") as file:
            initial_circuits = [line.strip() for line in file.readlines()]
        initial_circuits = np.array(initial_circuits)
        labels = np.vectorize(len)(initial_circuits)
        # 用index可同时控制 testset和labels
        indexs = np.arange(len(testset))

        try:
            while True:

                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file['data_buffer']
                            # print("buffer_size", self.buffer_size)
                            # print("size: data_buffer", len(self.data_buffer))
                            # print("size: batch_size", self.batch_size)
                            # self.iters = data_file['iters'] ## 数据被更新了多少次
                            del data_file
                        print('已载入数据')
                        break
                    except:
                        time.sleep(10)


                if len(self.data_buffer) > self.batch_size:
                    self.iter += 1
                    loss, entropy = self.network_updata()

                    ## 这里打印到tensorboard
                    writer = SummaryWriter('logs')
                    writer.add_scalar('total Loss', loss, self.iter)
                    writer.add_scalar('entropy', entropy, self.iter)
                    writer.close()




                time.sleep(CONFIG['train_update_interval'])  # 每1分钟更新一次模型

                if self.iter % self.check_freq == 0:
                    n_save = int(self.iter / self.check_freq)

                    ## 测试模块
                    lose = 0
                    # 抽取矩阵的index
                    num_to_select = len(indexs) // 2
                    selected = random.sample(indexs, num_to_select)

                    for i in selected:
                        target = testset[i]
                        circuit = QuantumCircuit(1)
                        len_cir = 0
                        for _ in range(CONFIG["maxstep"]):
                            len_cir += 1
                            act_probs, _ = self.policy_value_net(target)
                            act = np.argmax(act_probs)
                            if act == 0:
                                circuit.h(0)
                            elif act == 1:
                                circuit.s(0)
                            elif act == 2:
                                circuit.sdg(0)
                            elif act == 3:
                                circuit.t(0)
                            elif act == 4:
                                circuit.tdg(0)
                            circuit_unitary = Operator(circuit)
                            state = circuit_unitary.data
                            C_hs = 1 - np.abs(cirq.hilbert_schmidt_inner_product(state, target)) ** 2 / (4 ** CONFIG['n_qubit'])
                            if C_hs <= CONFIG["theta"]:
                                break
                        if len_cir > labels[i]:
                            lose += 1

                    accuary = 1 - lose/num_to_select
                    writer = SummaryWriter('logs')
                    writer.add_scalar('test performance ', accuary, n_save)
                    writer.close()

                    print("ith save model: {}".format(n_save))
                    # self.policy_value_net.save_model('models/models_history/current_policy_batch{}.model'.format(i + 1))
                    self.policy_value_net.save_model('models/current_policy.model')

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    training_pipeline = TrainPipeline(2)
    training_pipeline.run()