import numpy as np
import sympy as sp
from collections import deque
import copy
import cirq
from config import CONFIG
import MCTS
from network import prepare_input as pi

#全局变量
ket0 = np.array([[1], [0]]) # |0>
P0 = np.array([[1, 0], [0, 0]])  # |0><0|
P1 = np.array([[0, 0], [0, 1]])  # |1><1|


#基础门集 Clifford + T Gates
# I门
I = np.array([[1, 0], [0, 1]])
# Pauli-X 门 (X)
X = np.array([[0, 1], [1, 0]])
# Hadamard 门 (H)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
# 相位门 (S)
S = np.array([[1, 0], [0, 1j]])
# S 门的 Hermitian transpose
S_dagger = np.conj(S).T
# T 门
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
# T 门的 Hermitian transpose
T_dagger = np.conj(T).T
# CNOT 门 (Controlled-NOT 门)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
gate_Set = {"H": H, "S": S, "S_d": S_dagger, "T": T, "T_d": T_dagger, "C": CNOT, }



#生成电路初始状态
def get_state_initial(n_qubit):
    I = np.array([[1, 0], [0, 1]])
    for ith in range(n_qubit):
        if ith == 0:
            state = I
        else:
            state = np.kron(state, I)
    return state

#生成电路的合法操作 I⊗U⊗I
def get_gate_layer(n_qubit, gate, index): #2, "H", [1]
    I = np.array([[1, 0], [0, 1]])
    I_list = ["I" for _ in range(1,n_qubit+1)]
    gate_Set = {"I" : np.array([[1, 0], [0, 1]]),
                "X" : np.array([[0, 1], [1, 0]]),
        "H": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
                "S": np.array([[1, 0], [0, 1j]]),
                "S_d": np.array([[1, 0], [0, -1j]]),
                "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
                "T_d": np.array([[1, 0], [0, (np.sqrt(2) / 2) - 1j * (np.sqrt(2) / 2)]]),
                "C": np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]]),
                "P0" : np.array([[1, 0], [0, 0]]),# |0><0|
                "P1" : np.array([[0, 0], [0, 1]])
                }
    P0 = np.array([[1, 0], [0, 0]])  # |0><0|
    P1 = np.array([[0, 0], [0, 1]])  # |1><1|
    if len(index) == 1:  #single-bit
        list_tensor = copy.deepcopy(I_list)
        list_tensor[index[0]-1] = gate
        # for q in range(n_qubit):
        #     q += 1
        #     print(q)
        #     if q  == index:
        #         print("add gate")
        #         list_tensor.append(gate)
        #     else:
        #         print("add I")
        #         list_tensor.append(I)

        # 生成全局门矩阵
        if len(list_tensor) == 1:
            gate_layer = gate_Set[gate]
        else:
            for idx, item in enumerate(list_tensor):
               # print(item)
               if idx == 0:
                  gate_layer = gate_Set[item]
                  # print("item", item)
               else:
                  gate_layer = np.kron(gate_layer, gate_Set[item])

    elif len(index) == 2:  # CNOT
        # | 0 > < 0 | ⊗ I ⊗ I
        control_tensor = copy.deepcopy(I_list)
        control_tensor[index[0]-1] = "P0"
        # print("control_tesnor:", control_tensor)
        # print("I_list:", I_list)

        # | 1 > < 1 | ⊗ I ⊗ X
        target_tensor = copy.deepcopy(I_list)
        target_tensor[index[0]-1] = "P1"
        target_tensor[index[1]-1] = "X"
        # print("target_tesnor:", target_tensor)
        for idx, item in enumerate(control_tensor):
            # print(item)
            if idx == 0:
                control_layer = gate_Set[item]
                # print("item", item)
            else:
                control_layer = np.kron(control_layer, gate_Set[item])

        for idx, item in enumerate(target_tensor):
            # print(item)
            if idx == 0:
                target_layer = gate_Set[item]
                # print("item", item)
            else:
                target_layer = np.kron(target_layer, gate_Set[item])


        gate_layer = control_layer +  target_layer  # |0><0| ⊗ I ⊗ I + |1><1| ⊗ I ⊗ X

    return gate_layer

def index2gate(gate_Set):
    # actions_1bit = gate_Set + n_qubit -2
    i = 0
    ind_gate = {}
    for key, value in gate_Set.items():
        ind_gate[i] = key
        i += 1
    return ind_gate
def gen_dict_id2actions(id_list, num_gate_inset, num_qubit, dict_actions2id):
    dict_id2actions = {}
    num_gate_all = num_gate_inset + num_qubit - 2

    for id in id_list:
        con_bits = list(range(num_qubit))
        # print("con_bits", con_bits)
        target_bit = int(id // num_gate_all)
        # print("target_bit",target_bit)

        gate_index = int(id % num_gate_all)
        # print("gate_index", gate_index)
        if gate_index < num_gate_inset - 1:  #单量子比特门
            gate = dict_actions2id[gate_index]
            target_bit = [target_bit]
        else:       #CNOT门
            # con_bits = con_bits.remove(target_bit)
            gate = "C"
            # print(gate, "双比特门")
            # print("con_bits", con_bits)
            # print("target bit", target_bit)
            con_bits.remove(target_bit)
            # print("con_bits remove", con_bits)
            index = gate_index - (num_gate_inset - 1)
            # print("control bit index", index)
            # print("control bit list", con_bits)
            con_bit = con_bits[index]
            target_bit = [con_bit, target_bit]
        # print("gate",gate, target_bit)
        dict_id2actions[id] = [gate,target_bit]
    return dict_id2actions



class circuit(object):
    def __init__(self, n_qubit, gate_Set, target_matrix, theta):
        self.n_qubit = n_qubit
        self.state = None
        self.gate_Set = gate_Set
        self.gate_list = []
        self.target = target_matrix
        self.theta = theta
        self.all_actions_id = []
        self.dict_actions2id = index2gate(self.gate_Set)
        self.dict_id2actions = {}


    def init_circuit(self):
        self.state = get_state_initial(self.n_qubit)
        num_actions = len(self.gate_Set) + self.n_qubit -2
        self.all_actions_id = list(range(num_actions*self.n_qubit))
        self.dict_id2actions = gen_dict_id2actions(self.all_actions_id, len(self.gate_Set), self.n_qubit, self.dict_actions2id)



    def insert_gate(self, gate, index):
        gate_layer = get_gate_layer(self.n_qubit, gate, index)
        self.state = np.dot(gate_layer, self.state)
        self.gate_list.append([gate, index])


    # def get_all_actions(self):
    #     num_action = self.n_qubit * (len(self.gate_Set)- 1 + self.n_qubit - 1) #单量子比特+CNOT门
    #     return num_action


    def is_target(self):
        is_target = False
        product_HS = cirq.hilbert_schmidt_inner_product(self.target, self.state)
        # print("inner_product ", product_HS)
        # print("theta", self.theta)
        if np.abs(1 - product_HS) <= self.theta:
            is_target = True
        return is_target, product_HS




class Sys2target(object):
    def __init__(self, n_qubit, gate_Set, matrix_tar ):
        self.target = matrix_tar
        self.n_qubit = n_qubit
        self.gate_Set = gate_Set
        self.theta = CONFIG["theta"]
        self.actions_1bit = len(self.gate_Set) + self.n_qubit -2
        self.max_step = CONFIG["maxstep"]
        self.circuit = circuit(self.n_qubit, self.gate_Set, self.target,self.theta)
        self.index_gate = index2gate(self.gate_Set)
        self.num_action = self.actions_1bit * self.n_qubit

    def move_to_action(self, move):
        move  += 1
        n = self.actions_1bit
        bit =int(move // n)
        gate_index = int(move % n)
        if gate_index < len(self.gate_Set):
            gate = self.index_gate[gate_index]
        else:
            gate = "C"
            con = bit - len(self.gate_Set) + 1
            bit = [con ,bit]
        bit = [bit]

        return gate, bit


    def reward(self, product_HS): # Hilbert Schmidt inner product
        C_HS = 1 - (1 / (4 * self.n_qubit)) * np.abs(product_HS) ** 2
        return C_HS


    def self_play(self, player_MCTS, temp = 1e-3):
        # 初始化电路
        self.circuit.init_circuit()
        states, moves, probs_moves, rewards = [],[],[],[]
        #在达到最大步数时停止
        print("maxstep", self.max_step)
        for _ in range(self.max_step):

            move_mcts, probs_move_mcts = player_MCTS.get_action(self.circuit, self.num_action,temp = temp, return_prob = 1)
            print("move_mcts, probs_move_mcts", move_mcts, probs_move_mcts)
            #保存数据
            states.append(self.circuit.state)
            moves.append(move_mcts)
            probs_moves.append(probs_move_mcts)

            # id turn into gate
            action = self.circuit.dict_id2actions.get(move_mcts)
            print("action ", action)
            # 执行插入一个门
            self.circuit.insert_gate(action[0],action[1]) # move_mcts -> gate, index

            # Hilbert Schmidt inner product

            is_target,product_HS = self.circuit.is_target()
            #计算奖励，保存奖励
            reward_current = self.reward(product_HS)
            # reward_accumulated += reward_current
            rewards.append(reward_current)

            #判断是否合成目标电路
            if is_target:
                player_MCTS.reset_player()
                return is_target, zip(states, moves, probs_moves, rewards)

        return is_target, zip(states, moves, probs_moves, rewards)

    def get_observation(self):
        observation = np.dot(self.circuit.state, self.target)
        return observation
#可视化
def visualize_circuit(state):
    return None












if __name__ == '__main__':
    None
    # gate_layer = get_gate_layer(3,"C", [1,2])
    # print(gate_layer)

    # ini = get_state_initial(3)
    # print(ini)
    # print(np.eye(8,8))
    # circuit = circuit(2, gate_Set, None, None)
    # test1 = Sys2target(3,gate_Set,None,20)
    # print(test1.actions_1bit)
    # print(len(gate_Set))
    # print(test1.move_to_action(6))
    # print(circuit.get_all_actions())
    # print(index2gate(gate_Set))

    # dict_action_id = index2gate(gate_Set)
    # print(dict_action_id)
    # id_list = list(range(12))
    # print("id:", id_list)
    # print(gen_dict_id2actions(id_list,6,2,dict_action_id))
    #
    # res = get_state_initial(2)
    # print(res)


