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

def action_filter(action_pre, acts, probs, set_gate): #将一些不合理的action的概率变为0
    # print("action_filter:", set_gate)

    gate_pre = set_gate[action_pre]
    # print("action_filter: gate: {}, bite: {}".format(gate_pre[0], gate_pre[1]))
    if gate_pre[0] == "C" or gate_pre[0] == "H" or gate_pre[0] == "T" or gate_pre[0] == "T_d":
        probs[action_pre] = 0
    if gate_pre[0] == "T":
        # action_key1 = [key for key, value in set_gate.items() if value[0] == 'T_d'and value[1] == gate_pre[1] ]
        # print("action_key:", action_key1)
        probs[4] = 0  #prob_T_dg = 0
        # action_key2 = [key for key, value in set_gate.items() if value[0] == 'S_d'and value[1] == gate_pre[1] ]
        # print("action_key:", action_key2)
        probs[2] = 0 #prob_S_dg = 0
    elif gate_pre[0] == "T_d":
        # action_key1 = [key for key, value in set_gate.items() if value[0] == 'T' and value[1] == gate_pre[1]]
        # print("action_key:", action_key)
        probs[3] = 0 #prob_T = 0
        # action_key2 = [key for key, value in set_gate.items() if value[0] == 'S' and value[1] == gate_pre[1]]
        # print("action_key:", action_key)
        probs[1] = 0 #prob_S = 0
    elif gate_pre[0] == "S":
        # action_key1 = [key for key, value in set_gate.items() if value[0] == 'S_d' and value[1] == gate_pre[1]]
        # print("action_key:", action_key)
        probs[2] = 0 #prob_S_dg = 0
        # action_key2 = [key for key, value in set_gate.items() if value[0] == 'T_d' and value[1] == gate_pre[1]]
        # print("action_key:", action_key)
        probs[4] = 0  #prob_T_dg = 0
    elif gate_pre[0] == "S_d":
        # action_key1 = [key for key, value in set_gate.items() if value[0] == 'S' and value[1] == gate_pre[1]]
        # print("action_key:", action_key)
        probs[1] = 0 #prob_S = 0
        # action_key2 = [key for key, value in set_gate.items() if value[0] == 'T' and value[1] == gate_pre[1]]
        # print("action_key:", action_key)
        probs[3] = 0 #prob_T = 0

    return acts, probs


def probs_normalize(probs):
    total = np.sum(probs)
    probs /= total
    return probs


class Net(nn.Module):
    def __init__(self, input_dim = CONFIG["input_dim"], output_dim = CONFIG["output_dim"]):
        super(Net, self).__init__()
        # 公共网络
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer1_act = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.layer2_act = nn.ReLU()
        self.layer3 = nn.Linear(256, 256)
        self.layer3_act = nn.ReLU()

        # 策略头
        self.policy_layer = nn.Linear(256, output_dim)
        self.policy_act = nn.Softmax(dim=-1)

        # 价值头
        self.value_layer = nn.Linear(256, 1)
        self.value_act = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_act(x)
        x = self.layer2(x)
        x = self.layer2_act(x)
        x = self.layer3(x)
        x = self.layer3_act(x)

        # 策略头
        policy = self.policy_layer(x)
        policy = self.policy_act(policy)

        # 价值头
        value = self.value_layer(x)
        value = self.value_act(value)

        return policy, value




class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device = CONFIG["device"]):

            self.use_gpu = use_gpu
            self.l2_const = 2e-3  # l2 正则化
            self.device = device
            self.policy_value_net = Net().to(self.device)
            self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=CONFIG["learn_rate"], betas=(0.9, 0.999),
                                              eps=1e-8, weight_decay=self.l2_const)
            if model_file:
                # print("MODEL", model_file)
                self.policy_value_net.load_state_dict(torch.load(model_file)) # 加载模型参数 model_file：存储模型的地址
                # first_layer_weights_after = next(iter(self.policy_value_net.parameters())).detach().clone()
                # print("Weights after loading:", first_layer_weights_after)

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值，用在训练时候
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        # state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = log_act_probs.detach().numpy()
        return act_probs, value.detach().numpy()

    # 输入state，返回每个合法动作的（动作，概率）元组列表，以及state的value，用在预测
    def policy_value_fn(self, circuit):
        self.policy_value_net.eval()
        state_dg = circuit.state.conj().T  #V†
        state_input = np.dot(state_dg, circuit.target) # V†U represent unbuilt part

        # 获取合法动作列表
        # legal_positions = circuit.all_actions_id  #action masking
        # state_input = prepare_input(get_observation(circuit))
        # state_input = np.ascontiguousarray(state_input)
        # print(state_input)

        state_input = torch.as_tensor(state_input).to(self.device)
        # print("state_input", state_input)
        real_part = state_input.real  # 提取实部
        imag_part = state_input.imag  # 提取虚部
        # print(real_part)
        # print(imag_part)
        # 展平实部和虚部
        flattened_real = real_part.flatten()
        flattened_imag = imag_part.flatten()
        # 连接实部和虚部
        # print(flattened_real)
        # print(flattened_imag)

        state_input = torch.cat((flattened_real, flattened_imag), dim=0).to(torch.half)  #flatten
        # state_batch = prepare_inputs(state_batch)
        # state_input = torch.stack([state_input], dim=0)

        # 使用神经网络进行预测
        with autocast(): #转换为半精度fp16
            # print("state_input", state_input)
            log_act_probs, value = self.policy_value_net(state_input)  #调用前向传播函数
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = log_act_probs.detach().numpy().astype('float16').flatten()

        actions = np.arange(len(circuit.gate_Set))

        if circuit.gate_inserted:
           ## 过滤
            gate_pre = circuit.gate_inserted[-1]
            actions, act_probs = action_filter(gate_pre, actions,act_probs, circuit.gate_Set)
            act_probs = probs_normalize(act_probs)  #probs normalize



        act_probs = zip(actions, act_probs[actions])
        # act_probs = zip(actions_legal, act_probs[actions_legal])  # 合法action未归一化
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


