# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import esm

AAS = "ILVAGMFYWEDQNHCRKSTP"


def states_to_seqs(states):
    return ''.join([AAS[i] for i in torch.argmax(states, dim=0).cpu().numpy()])


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # action policy layers
        # self.act_module = nn.Sequential(
        #     nn.Linear(
        #         self.board_width * 1280, self.board_width * self.board_height
        #     ),
        #     nn.ReLU(),
        #     nn.Linear(
        #         self.board_width * self.board_height,
        #         self.board_width * self.board_height
        #     ),
        # )
        self.act_fc1 = nn.Linear(1280, board_height * 4)
        self.act_fc2 = nn.Linear(
            board_width * board_height * 4, board_width * board_height
        )
        # state value layers
        # self.val_module = nn.Sequential(
        #     nn.Linear(self.board_width * 1280, self.board_width * 64),
        #     nn.ReLU(),
        #     nn.Linear(self.board_width * 64, 1),
        # )
        self.val_fc1 = nn.Linear(1280, board_height)
        self.val_fc2 = nn.Linear(board_width * board_height, 128)
        self.val_fc3 = nn.Linear(128, 1)

    def forward(self, token_representations):
        # action policy layers
        # x_act = self.act_module(token_representations)
        x_act = self.act_fc1(token_representations)
        x_act = F.relu(x_act).view(-1, self.board_width * self.board_height * 4)
        x_act = self.act_fc2(x_act)
        x_act = F.log_softmax(x_act, dim=-1)
        # state value layers
        # x_val = F.tanh(self.val_module(token_representations))
        x_val = self.val_fc1(token_representations)
        x_val = F.relu(x_val).view(-1, self.board_width * self.board_height)
        x_val = self.val_fc2(x_val)
        x_val = F.relu(x_val)
        x_val = self.val_fc3(x_val)
        x_val = F.tanh(x_val)
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(
        self,
        board_width,
        board_height,
        feature_extractor,
        model_file=None,
        use_gpu=False
    ):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        # the policy value net module
        self.feature_extractor = feature_extractor
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.AdamW(
            [
                params for params in self.policy_value_net.parameters()
                if params.requires_grad
            ],
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = torch.tensor(state_batch).cuda().permute(0, 2, 1)
            features = self.feature_extractor(state_batch)
            log_act_probs, value = self.policy_value_net(features)
            act_probs = torch.exp(log_act_probs)
            return act_probs, value
        else:
            state_batch = torch.tensor(state_batch).permute(0, 2, 1)
            features = self.feature_extractor(state_batch)
            log_act_probs, value = self.policy_value_net(features)
            act_probs = torch.exp(log_act_probs)
            return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state_0 = np.expand_dims(board.current_state(),
                                         axis=0).transpose(0, 2, 1)
        current_state = np.ascontiguousarray(current_state_0)  ##

        if self.use_gpu:
            features = self.feature_extractor(
                Variable(torch.from_numpy(current_state)).cuda().float()
            )
            log_act_probs, value = self.policy_value_net(features)
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            features = self.feature_extractor(
                Variable(torch.from_numpy(current_state)).float()
            )
            log_act_probs, value = self.policy_value_net(features)
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = torch.tensor(state_batch).cuda().permute(0, 2, 1)
            mcts_probs = torch.tensor(mcts_probs).cuda()
            winner_batch = torch.tensor(winner_batch).cuda()
        else:
            state_batch = torch.tensor(state_batch).permute(0, 2, 1)
            mcts_probs = torch.tensor(mcts_probs)
            winner_batch = torch.tensor(winner_batch)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        features = self.feature_extractor(state_batch)
        log_act_probs, value = self.policy_value_net(features)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
