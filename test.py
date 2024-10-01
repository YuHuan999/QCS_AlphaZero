import torch
import env_QCS as env
import network
if __name__ == '__main__':

    # log_act_probs = torch.tensor([-2.0, -1.0, -0.5, -1.5, -1.1, -0.9, -2.2, -0.7, -1.3, -2.1, -0.8, -1.6])
    # act_probs = torch.exp(log_act_probs.detach().numpy())
    # # print(act_probs)
    # state_batch = torch.tensor(state_batch).to(self.device)
    # log_act_probs, value = self.policy_value_net(state_batch)
    # log_act_probs, value = log_act_probs.cpu(), value.cpu()
    # act_probs = np.exp(log_act_probs.detach().numpy())
    # return act_probs, value.detach().numpy()
    # for action, prob in act_probs:
    # #     print(action, prob)
    # print(torch.cuda.is_available())
    # RES = env.get_gate_layer(1, "H", [1])
    # print(RES)


    try:
        net = network.PolicyValueNet(r'models/current_policy.model', use_gpu=True, device = 'cpu')
        # data = torch.load(r'models/current_policy.model')
        print("模型加载成功")
    except Exception as e:
        print("加载模型时发生错误：", str(e))