# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of EvoPlay for PAB1 protein mutation

@author: Yi Wang
"""

from __future__ import print_function

import sys
import random
import pathlib
import datetime
from typing import List, Union
from collections import defaultdict, deque

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

import esm
from env_model import ESM_predictor
from p_v_net_esm import PolicyValueNet  # Pytorch
from residue_constant import AAS
from sequence_env_m_p import Seq_env, Mutate
from mcts_alphaZero_mutate_expand_m_p_gfp import MCTSMutater

data_path = 'code/evoplay/data/PAB1_GFP_data/PAB1.txt'

pab1_wt_sequence = (
    "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP"
)
starts = {
    "start_seq":
        "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALKGMLLNGQEIYFAP"  # noqa: E501
}


def string_to_one_hot(sequence: str, alphabet: str) -> torch.Tensor:
    out = torch.tensor([AAS.index(residue) for residue in sequence])
    out = torch.nn.functional.one_hot(out, num_classes=len(alphabet))
    return out.float()


def raw_to_features(data_dir):
    with open(data_dir, "r") as fr:
        ll = fr.readlines()
        seq_list = []
        label_list = []
        for i in range(1, len(ll)):
            tmp_list = ll[i].strip().split("\t")
            seq_list.append(tmp_list[0])
            label_list.append(float(tmp_list[1]))

    labels = torch.tensor(label_list, dtype=torch.float32)
    one_hots = torch.stack([string_to_one_hot(seq, AAS) for seq in seq_list]
                          ).to(torch.float32)
    return one_hots, labels


def train_surrogate_predictor(data_dir, feature_extractor, retrain=False):
    print("training score predictor")
    one_hots, labels = raw_to_features(data_dir)
    seq_dataset = MyDataset(one_hots, labels)
    epochs = 5
    train_loader = DataLoader(seq_dataset, batch_size=512, shuffle=True)

    model = ESM_predictor(
        len(pab1_wt_sequence),
        len(AAS),
    ).cuda()
    if pathlib.Path('out/PAB1_GFP_task/PAB1/surrogate_predictor_esm.pth'
                   ).exists():
        print("Loading existing surrogate predictor model...")
        model.load_state_dict(
            torch.load('out/PAB1_GFP_task/PAB1/surrogate_predictor_esm.pth')
        )
        print("Model loaded successfully.")
    if retrain:
        optimizer = optim.AdamW(
            [params for params in model.parameters() if params.requires_grad],
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                inputs = batch[0].cuda()
                labels = batch[1].cuda()
                optimizer.zero_grad()

                features = feature_extractor(inputs)
                features = features.mean(1)
                logits = model(features)
                loss = F.mse_loss(logits, labels)
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch}, train_loss {loss}")
        pathlib.Path('out/PAB1_GFP_task/PAB1'
                    ).mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            "out/PAB1_GFP_task/PAB1/surrogate_predictor_esm.pth"
        )
        print("Surrogate predictor trained and saved.")
    torch.cuda.empty_cache()
    return model


def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


class FeatureExtracter(torch.nn.Module):
    def __init__(self):
        super(FeatureExtracter, self).__init__()
        print("Loading ESM2 model...")
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm_model.eval()
        self.esm_model.requires_grad_(False)
        print("ESM2 model loaded.")

    def _states_to_seqs(self, states):
        return ''.join([AAS[i] for i in torch.argmax(states, dim=-1).tolist()])

    def forward(self, state_input):
        data = [
            ("seq", self._states_to_seqs(state_input_one))
            for state_input_one in state_input
        ]
        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.esm_model(
                batch_tokens.cuda(), repr_layers=[33], return_contacts=False
            )
            token_representations = results["representations"][
                33][:, 1:-1, :].clone().detach()
        return token_representations


class MyDataset(data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, index):
        seq, target = self.sequences[index], self.labels[index]
        return seq, target

    def __len__(self):
        return len(self.sequences)


class TrainPipeline():
    def __init__(
        self,
        start_seq,
        alphabet,
        model,
        feature_extractor,
        trust_radius,
        init_model=None
    ):  #init_model=None
        self.seq_len = len(start_seq)
        self.vocab_size = len(alphabet)
        self.n_in_row = 4
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            feature_extractor,
            model,
            start_seq,
            trust_radius
        )  # n_in_row=self.n_in_row
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 64  # num of simulations for each move 400 1600
        self.c_puct = 10  #0.5  # 10
        self.buffer_size = 10000
        self.batch_size = 32  # mini-batch size for training  512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 64
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        # self_added
        self.buffer_no_extend = False
        # self_added
        # playout
        self.generated_seqs = []
        self.fit_list = []
        self.p_dict = {}
        self.m_p_dict = {}
        self.retrain_flag = False
        self.part = 2
        # playout
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(
                self.seq_len,
                self.vocab_size,
                feature_extractor,
                model_file=init_model,
                use_gpu=True
            )
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(
                self.seq_len, self.vocab_size, feature_extractor, use_gpu=True
            )
        self.mcts_player = MCTSMutater(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        counts = len(self.generated_seqs)
        self.buffer_no_extend = False
        for i in range(n_games):
            play_data, seq_and_fit, p_dict = self.mutate.start_mutating(
                self.mcts_player, temp=self.temp)  #winner,
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            self.p_dict = p_dict
            self.m_p_dict.update(self.p_dict)
            if self.episode_len == 0:
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                for seq, fit in seq_and_fit:  #alphafold_d
                    if seq not in self.generated_seqs:
                        self.generated_seqs.append(seq)
                        self.fit_list.append(fit)
                        if seq not in self.m_p_dict.keys():
                            self.m_p_dict[seq] = fit

                        if len(self.generated_seqs) % 10 == 0 and len(
                            self.generated_seqs
                        ) > counts and self.part <= 10:
                            self.retrain_flag = True

    def policy_update(self):
        """update the policy-value net"""
        for i in range(self.epochs):
            mini_batch = random.sample(self.data_buffer, k=self.batch_size)
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]
            old_probs, old_v = self.policy_value_net.policy_value(state_batch)
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = torch.mean(
                torch.sum(
                    old_probs * (
                        torch.log(old_probs + 1e-10) -
                        torch.log(new_probs + 1e-10)
                    ),
                    axis=1
                )
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (
            1 - torch.var(torch.tensor(winner_batch).cuda() - old_v.flatten()) /
            torch.var(torch.tensor(winner_batch))
        )
        explained_var_new = (
            1 - torch.var(torch.tensor(winner_batch).cuda() - new_v.flatten()) /
            torch.var(torch.tensor(winner_batch))
        )
        print(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new
            )
        )
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        starttime = datetime.datetime.now()
        print("part:2")
        #part = 2
        try:
            for i in range(self.game_batch_num):
                print(f"batch {i + 1} starts")
                self.collect_selfplay_data(self.play_batch_size)
                print(f"batch {i + 1}, episode_len:{self.episode_len}")
                print(f"batch {i + 1}, data_buffer len:{len(self.data_buffer)}")
                # if (i + 1) % 16 == 0:
                #     print('train predictor again')
                #     update_model = train_surrogate_predictor(
                #         data_path, feature_extractor, retrain=False
                #     )
                #     self.seq_env.model = update_model
                #     self.seq_env.model.eval()
                #     self.retrain_flag = False
                if self.buffer_no_extend == False and len(
                    self.data_buffer
                ) >= self.batch_size and (i + 1) != self.game_batch_num:
                    print("start policy update")
                    loss, entropy = self.policy_update()
                print()
            m_p_fitness = np.array(
                [i.cpu().numpy() for i in self.m_p_dict.values()]
            )
            m_p_seqs = np.array(list(self.m_p_dict.keys()))
            df_m_p = pd.DataFrame(
                {
                    "sequence": m_p_seqs, "pred_fit": m_p_fitness
                }
            )
            pathlib.Path('out/PAB1_GFP_task/PAB1/generate'
                        ).mkdir(parents=True, exist_ok=True)
            df_m_p.to_csv(
                "out/PAB1_GFP_task/PAB1/generate/evoplay_pab1_generated_sequence_esm_5.csv",
                index=False
            )
            endtime = datetime.datetime.now()
            print('time cost:', (endtime - starttime).seconds)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    feature_extractor = FeatureExtracter().eval().cuda()
    model = train_surrogate_predictor(
        data_path, feature_extractor, retrain=False
    )
    training_pipeline = TrainPipeline(
        starts["start_seq"],
        AAS,
        model,
        feature_extractor,
        trust_radius=100,
    )
    training_pipeline.run()
