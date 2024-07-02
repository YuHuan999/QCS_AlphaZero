import numpy as np
import copy
from config import CONFIG


def softmax(x):
    probs = np.exp(x-np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):

    def __init__(self, parent, probs_piror):
        # parent: parent node
        # probs_piror: the probability of select this node
        self.parent = parent
        self.p = probs_piror
        self.children = {}
        self.count_visit = 0 # 当前节点被访问次数
        self.Q = 0 # 当前节点的对应动作的平均动作价值
        self.U = 0 # 当前节点的对应动作的置信上界 PUCT

    def expand(self, action_prob): # 非法动作概率为0
        #创建新的子节点 来展开树
        for action, prob in action_prob:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob) # self：parent


    def get_value(self,c_puct):
        #c_puct：控制相对影响 大：倾向探索 小：倾向Q值
        self.U = (c_puct*self.p* np.sqrt(self.parent.count_visit/(1 + self.count_visit) ) )
        return self.Q + self.U  #节点先验U和评估Q


    def select(self, c_puct):
        #在子节点中选择最大的Q+U
        # return ：(action, node)
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))


    def update_recursive(self, leaf_value, rewards):
        if self.parent:
            reward = rewards.pop() if rewards else 0  # 确保有奖励可用
            # q(t) = r +  q(t+1)* alpha
            Qs = CONFIG["alpha"] * (reward + leaf_value) + self.parent.Q * (self.parent.count_visit - 1)
            self.parent.count_visit += 1
            self.parent.Q = Qs / self.parent.count_visit
            # 更新当前的leaf_value为父节点的Q值
            leaf_value = self.parent.Q
            self.parent.update_recursive(leaf_value, rewards)              #递归


    def is_leave(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS(object):

    def __init__(self, polic_value_fn, c_puct = 5, n_playout = 2000):
    #polic_value_fn:神经网络提供:接受当前状态-》返回接下来的动作分布和当前状态的评分
        self.root = TreeNode(None, 1.0)
        self.polic_value = polic_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def playout(self, circuit):
        #state: 当前电路的状态
        node = self.root #从根节点开始 作为一次playout
        rewards = []
        while True:
            if node.is_leave():
                break
            #贪心算法选择最大置信上界
            action, node = node.select(self.c_puct)
            action = circuit.dict_id2actions[action]
            circuit.insert_gate(action[0], action[1]) # action代码 转换为 门操作 insert_gate 可识别
            product_HS = circuit.is_target()[1]
            reward = 1 - (1 / (4 * CONFIG["n_qubit"])) * np.abs(product_HS) ** 2
            rewards.append(reward)

        # state = circuit.state # current node is leave
        # print("state_MCTS ")
        #如果是叶节点 给出由神经网络提供的 所有可能的动作分布以及对当前环境的评估
        #给出动作的概率 分理出动作：概率
        actions_probs, leaf_value = self.polic_value(circuit) # actions[0.1353, 0.3679, 0.6065, 0.2231,0.1225, 0.4493, 0.2019]] and evaluated q(s+1) of current node
        #看电路是否满足合成要求：
        is_target, product_HS= circuit.is_target()
        reward = 1 - (1 / (4 * CONFIG["n_qubit"])) * np.abs(product_HS) ** 2  #当前step的reward或者cost
        # print("is_target", is_target)
        if not is_target:
            node.expand(actions_probs)
            node.Q = leaf_value
        else:
            node.Q = reward
        node.count_visit += 1

        #更新节点值访问次数
        node.update_recursive(node.Q, rewards)

    def get_move_probs(self, circuit, tem=1e-3 ):
        # 获得当前状态的action的蒙特卡洛分布
        # state: 当前游戏状态
        # tem：温度 （0,1】

        for n in range(self.n_playout):
            circuit_copy = copy.deepcopy(circuit)
            self.playout(circuit_copy)


        actions_visit = [(act, node.count_visit) for act, node in self.root.children.items()]
        # print("node",self.root, self.root.children.items())
        acts, visit = zip(*actions_visit)

        action_probs = softmax(1.0/ tem * np.log(np.array(visit) + 1e-10))
        return acts, action_probs

    def update_root(self, move_last):
        #实际地向前扩展 根据蒙特卡洛树的分布结果 在游戏中走一步
        if move_last in self.root.children:
            self.root = self.root.children[move_last]
            self.parent = None

        else:
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'

class MCTSplayer(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    #重置搜索树
    def reset_player(self):
        self.mcts.update_root(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    def get_action(self, circuit, num_actions, temp=1e-3, return_prob=0):
        # 像alphaGo_Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(num_actions) #所有actions
        # state = circuit.state
        acts, probs = self.mcts.get_move_probs(circuit, temp)
        move_probs[list(acts)] = probs

        if self._is_selfplay:
            # 添加Dirichlet Noise进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )

            # 更新根节点并重用搜索树
            self.mcts.update_root(-1)

        if return_prob:
            return move, move_probs
        else:
            return move








