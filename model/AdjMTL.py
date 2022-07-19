# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Expert_shared, Expert_task1, Expert_task2 are similar. Just a Linear(in, out)
class Expert_net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Expert_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)
        # init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        return self.fc1(x)


# Gate_shared, Gate_task1, Gate_task2 are similar. Just a Linear(in, out)
class Gate_net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Gate_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)
        # init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        return self.fc1(x)


class GatingNetwork(nn.Module):

    def __init__(self, input_units, units, num_experts, selectors, gate_units, base=True):
        super(GatingNetwork, self).__init__()

        self.ui_units = (input_units // 3) * 2
        self.a_units = input_units // 3
        self.units = units
        self.num_expers = num_experts
        self.base = base

        self.experts_shared = nn.ModuleList([Expert_net(input_units, units) for _ in range(num_experts)])
        self.experts_task1 = nn.ModuleList([Expert_net(input_units, units) for _ in range(num_experts)])
        self.experts_task2 = nn.ModuleList([Expert_net(input_units, units) for _ in range(num_experts)])
        self.expert_activation = nn.ReLU()

        self.gate_shared = Gate_net(input_units, num_experts * 3)
        self.gate_task1 = Gate_net(input_units, selectors * num_experts)  # Wk, d*m1
        self.gate_task2 = Gate_net(input_units, selectors * num_experts)  # Wk, d*m1

        self.gate_share_2 = Gate_net(gate_units, num_experts)
        self.gate_task2_ip = Gate_net(gate_units, num_experts)
        self.gate_task2_up = Gate_net(gate_units, num_experts)

        self.gate_share_1_ip = Gate_net(gate_units, num_experts)
        self.gate_share_1_up = Gate_net(gate_units, num_experts)
        self.gate_task1_ui = Gate_net(gate_units, num_experts)
        self.gate_activation = nn.Softmax(dim=-1)

        self.headW = nn.Linear(3 * units, units)
        self.alpha = 0.1
        self.beta = 0.1

    def forward(self, gate_output_shared_final, gate_output_task1_final, gate_output_task2_final, uikey, upkey, ipkey):
        # ui ui x
        # expert shared
        expert_shared_o = [e(gate_output_shared_final) for e in self.experts_shared]    # [bs*out,...,bs*out]
        expert_shared_tensors = torch.cat(expert_shared_o, dim=0)                       # bs * (num_expert * out)
        expert_shared_tensors = expert_shared_tensors.view(-1, self.num_expers, self.units)     # bs * num_expert * out
        expert_shared_tensors = self.expert_activation(expert_shared_tensors)
        # expert task1
        expert_task1_o = [e(gate_output_task1_final) for e in self.experts_task1]
        expert_task1_tensors = torch.cat(expert_task1_o, dim=0)
        expert_task1_tensors = expert_task1_tensors.view(-1, self.num_expers, self.units)
        expert_task1_tensors = self.expert_activation(expert_task1_tensors)
        # expert task2
        expert_task2_o = [e(gate_output_task2_final) for e in self.experts_task2]
        expert_task2_tensors = torch.cat(expert_task2_o, dim=0)
        expert_task2_tensors = expert_task2_tensors.view(-1, self.num_expers, self.units)
        expert_task2_tensors = self.expert_activation(expert_task2_tensors)

        # gate task1
        # get gate vector
        gate_output_task1 = self.gate_task1(gate_output_task1_final)    # bs * num_expert
        gate_output_task1 = self.gate_activation(gate_output_task1)     # bs * softmax(num_expert)
        # 将gate向量和expert的输出相乘
        gate_expert_output1 = torch.cat([expert_shared_tensors, expert_task1_tensors], dim=1)   # bs * 2num_expert * out
        gate_output_task1 = torch.einsum('be,beu ->beu', gate_output_task1, gate_expert_output1)
        gate_output_task1 = gate_output_task1.sum(dim=1)   # bs * out

        # Res gate task1

        gate_output_share_1_ip = self.gate_share_1_ip(ipkey)   # match by (u,i), to extract information in shared expert
        gate_output_share_1_ip = self.gate_activation(gate_output_share_1_ip)
        gate_output_share_1_up = self.gate_share_1_up(upkey)
        gate_output_share_1_up = self.gate_activation(gate_output_share_1_up)
        gate_output_task1_ui = self.gate_task1_ui(uikey)
        gate_output_task1_ui = self.gate_activation(gate_output_task1_ui)

        gate_expert_share_1_ip = torch.einsum('be,beu ->beu', gate_output_share_1_ip, expert_shared_tensors)
        gate_expert_share_1_ip = gate_expert_share_1_ip.sum(dim=1)
        gate_expert_share_1_up = torch.einsum('be,beu ->beu', gate_output_share_1_up, expert_task2_tensors)
        gate_expert_share_1_up = gate_expert_share_1_up.sum(dim=1)
        gate_expert_task1_ui = torch.einsum('be,beu ->beu', gate_output_task1_ui, expert_task2_tensors)
        gate_expert_task1_ui = gate_expert_task1_ui.sum(dim=1)

        gate_output_task1 = gate_output_task1 + self.alpha * gate_expert_share_1_ip \
                            + self.alpha * gate_expert_share_1_up + self.alpha * gate_expert_task1_ui
        # gate_output_task1 = 0.25 * gate_output_task1 + 0.25 * gate_expert_share_1_ip + 0.25 * gate_expert_share_1_up + 0.25 * gate_expert_task1_ui

        # gate task2
        gate_output_task2 = self.gate_task2(gate_output_task2_final)  # bs * num_expert
        gate_output_task2 = self.gate_activation(gate_output_task2)  # bs * softmax(num_expert)

        gate_expert_output2 = torch.cat([expert_shared_tensors, expert_task2_tensors], dim=1)  # bs * 2num_expert * out
        gate_output_task2 = torch.einsum('be,beu ->beu', gate_output_task2, gate_expert_output2)
        gate_output_task2 = gate_output_task2.sum(dim=1)  # bs * out

        # Res gate task2

        gate_output_share_2 = self.gate_share_2(uikey)
        gate_output_share_2 = self.gate_activation(gate_output_share_2)
        gate_output_task2_ip = self.gate_task2_ip(ipkey)
        gate_output_task2_ip = self.gate_activation(gate_output_task2_ip)
        gate_output_task2_up = self.gate_task2_up(upkey)
        gate_output_task2_up = self.gate_activation(gate_output_task2_up)

        gate_expert_share_2 = torch.einsum('be,beu ->beu', gate_output_share_2, expert_shared_tensors)
        gate_expert_share_2 = gate_expert_share_2.sum(dim=1)
        gate_expert_task2_ip = torch.einsum('be,beu ->beu', gate_output_task2_ip, expert_task2_tensors)
        gate_expert_task2_ip = gate_expert_task2_ip.sum(dim=1)
        gate_expert_task2_up = torch.einsum('be,beu ->beu', gate_output_task2_up, expert_task2_tensors)
        gate_expert_task2_up = gate_expert_task2_up.sum(dim=1)

        # if not self.base:
        gate_output_task2 = gate_output_task2 + self.beta * gate_expert_share_2 \
                            + self.beta * gate_expert_task2_up + self.beta * gate_expert_task2_ip
        # gate_output_task2 = 0.25 * gate_output_task2 + 0.25 * gate_expert_share_2 + 0.25 * gate_expert_task2_up + 0.25 * gate_expert_task2_ip
        # gate_output_task2 = self.headW(torch.cat([gate_expert_share_2, gate_expert_task2_up, gate_expert_task2_ip], dim=1))

        # gate shared
        gate_output_shared = self.gate_shared(gate_output_shared_final)
        gate_output_shared = self.gate_activation(gate_output_shared)

        gate_expert_output_shared = torch.cat([expert_task1_tensors, expert_shared_tensors, expert_task2_tensors], dim=1)

        gate_output_shared = torch.einsum('be,beu ->beu', gate_output_shared, gate_expert_output_shared)
        gate_output_shared = gate_output_shared.sum(dim=1)

        return gate_output_shared, gate_output_task1, gate_output_task2


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # init.xavier_normal_(self.fc2.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        out = torch.sigmoid(out)
        return out


class AdjMTLModel(nn.Module):

    def __init__(self, num_feature, h_units, num_experts, selectors, tower_h):
        super(AdjMTLModel, self).__init__()

        self.gate1 = GatingNetwork(num_feature, h_units, num_experts, selectors, (num_feature//3)*2)
        self.gate2 = GatingNetwork(h_units, h_units, num_experts, selectors, (num_feature//3)*2, base=False)

        self.towers = nn.ModuleList([
            Tower(h_units, 1, tower_h) for _ in range(2)    # 2为两个任务
        ])

    def forward(self, x):
        bs, dim = x.shape
        ui, ip = x[:, 0:(dim//3)*2], x[:, dim//3:]
        up = torch.cat([x[:, 0:dim//3], x[:, (dim//3)*2:]], dim=1)
        gate_output_shared, gate_output_task1, gate_output_task2 = self.gate1(x, x, x, ui, up, ip)
        _, task1_o, task2_o = self.gate2(gate_output_shared, gate_output_task1, gate_output_task2, ui, up, ip)

        final_output = [tower(task) for tower, task in zip(self.towers, [task1_o, task2_o])]

        return final_output