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
        self.act_module = nn.Sequential(
            nn.Linear(
                self.board_width * 1280, self.board_width * self.board_height
            ),
            nn.ReLU(),
            nn.Linear(
                self.board_width * self.board_height,
                self.board_width * self.board_height
            ),
        )
        # state value layers
        self.val_module = nn.Sequential(
            nn.Linear(self.board_width * 1280, self.board_width * 64),
            nn.ReLU(),
            nn.Linear(self.board_width * 64, 1),
        )

    def forward(self, token_representations):
        # action policy layers
        x_act = self.act_module(token_representations)
        x_act = F.log_softmax(x_act, dim=-1)
        # state value layers
        x_val = F.tanh(self.val_module(token_representations))
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
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            features = self.feature_extractor(state_batch).view(
                -1, self.board_width * 1280
            )
            log_act_probs, value = self.policy_value_net(features)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            features = self.feature_extractor(state_batch).view(
                -1, self.board_width * 1280
            )
            log_act_probs, value = self.policy_value_net(features)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state_0 = np.expand_dims(board.current_state(), axis=0)
        current_state = np.ascontiguousarray(current_state_0)  ##

        if self.use_gpu:
            features = self.feature_extractor(
                Variable(torch.from_numpy(current_state)).cuda().float()
            ).view(-1, self.board_width * 1280)
            log_act_probs, value = self.policy_value_net(features)
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            features = self.feature_extractor(
                Variable(torch.from_numpy(current_state)).float()
            ).view(-1, self.board_width * 1280)
            log_act_probs, value = self.policy_value_net(features)
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        features = self.feature_extractor(state_batch
                                         ).view(-1, self.board_width * 1280)
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
