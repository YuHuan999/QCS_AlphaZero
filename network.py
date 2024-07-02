import torch
import torch.nn as nn
from config import CONFIG
import numpy as np
import torch.nn.functional as F
from config import CONFIG
from torch.cuda.amp import autocast

#将复数矩阵，分成实数矩阵和虚数矩阵，然后铺平连接作为神经网络输入
def prepare_input(matrix):
    real_part = matrix.real
    imag_part = matrix.imag
    flattened_real = real_part.flatten()
    # print("f",flattened_real)
    flattened_imag = imag_part.flatten()
    # print("f",flattened_imag)
    input = torch.cat((torch.tensor(flattened_real, dtype=torch.float32), torch.tensor(flattened_imag, dtype=torch.float32)), dim=-1)
    return input
def get_observation(circuit):
    # print("state", circuit.state)
    # print("target", circuit.target)
    # print("circuit.state, type", circuit.state, type(circuit.state))
    observation = np.dot(circuit.state, circuit.target)
    return observation
class Net(nn.Module):
    def __init__(self, input_dim = 32, output_dim = 12):
        super(Net, self).__init__()
        # 公共网络
        self.layer1 = nn.Linear(input_dim, 1024)
        self.layer1_norm = nn.LayerNorm(1024)
        self.layer1_act = nn.ReLU()
        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_norm = nn.LayerNorm(1024)
        self.layer2_act = nn.ReLU()
        self.layer3 = nn.Linear(1024, 1024)
        self.layer3_norm = nn.LayerNorm(1024)
        self.layer3_act = nn.ReLU()
        self.layer4 = nn.Linear(1024, 1024)
        self.layer4_norm = nn.LayerNorm(1024)
        self.layer4_act = nn.ReLU()
        self.layer5 = nn.Linear(1024, 1024)
        self.layer5_norm = nn.LayerNorm(1024)
        self.layer5_act = nn.ReLU()

        # 策略头
        self.policy_layer = nn.Linear(1024, output_dim )
        self.policy_norm = nn.LayerNorm(output_dim )
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(output_dim , output_dim)

        # 价值头
        self.value_layer = nn.Linear(1024, 128)
        self.value_norm = nn.LayerNorm(128)
        self.value_act = nn.ReLU()
        self.value_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)
        x = self.layer2(x)
        x = self.layer2_norm(x)
        x = self.layer2_act(x)
        x = self.layer3(x)
        x = self.layer3_norm(x)
        x = self.layer3_act(x)
        x = self.layer4(x)
        x = self.layer4_norm(x)
        x = self.layer4_act(x)
        x = self.layer5(x)
        x = self.layer5_norm(x)
        x = self.layer5_act(x)

        # 策略头
        policy = self.policy_layer(x)
        policy = self.policy_norm(policy)
        policy = self.policy_act(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=-1)

        # 价值头
        value = self.value_layer(x)
        value = self.value_norm(value)
        value = self.value_act(value)
        value = self.value_fc(value)
        # value = torch.tanh(value)

        return policy, value


# 策略价值网络 用来训练
class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device = 'cpu'):

            self.use_gpu = use_gpu
            self.l2_const = 2e-3  # l2 正则化
            self.device = device
            self.policy_value_net = Net().to(self.device)
            self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                              eps=1e-8, weight_decay=self.l2_const)
            if model_file:
                self.policy_value_net.load_state_dict(torch.load(model_file))  # 加载模型参数 model_file：存储模型的地址

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        # state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self, circuit):
        self.policy_value_net.eval()
        # state = circuit.state

        # 获取合法动作列表
        legal_positions = circuit.all_actions_id  #action masking
        state_input = prepare_input(get_observation(circuit))
        state_input = np.ascontiguousarray(state_input)
        # print(state_input)
        state_input = torch.as_tensor(state_input).to(self.device)
        # print("state input, type; in network", state_input, type(state_input))

        # 使用神经网络进行预测
        with autocast(): #转换为半精度fp16
            log_act_probs, value = self.policy_value_net(state_input)  #调用前向传播函数
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        # 只取出合法动作
        act_probs = zip(legal_positions, act_probs[legal_positions])  # 合法action未归一化
        # 返回动作概率，以及状态价值
        return act_probs, value.detach().numpy()


    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)


    def train_step(self, state_batch, mcts_probs, value_batch, lr=0.002): ## winner_batch 肯定要修改 这里应该是state value
        self.policy_value_net.train()
        # 包装变量
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        value_batch = torch.tensor(value_batch).to(self.device)

        # 清零梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for params in self.optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
            params['lr'] = lr
        # 给出评估值
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value, target=value_batch)
        # 策略损失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))  # 希望两个向量方向越一致越好
        # 总的损失，注意l2惩罚已经包含在优化器内部
        loss = value_loss + policy_loss
        # 反向传播及优化
        loss.backward()
        self.optimizer.step()
        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()




if __name__ == '__main__':
    None


