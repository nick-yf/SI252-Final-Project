# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of EvoPlay for PAB1 protein mutation

@author: Yi Wang
"""

from __future__ import print_function
import pathlib
import random
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sequence_env_m_p import Seq_env, Mutate
from mcts_alphaZero_mutate_expand_m_p_gfp import MCTSMutater
# from p_v_net_torch import PolicyValueNet  # Pytorch
from p_v_net_esm import PolicyValueNet  # Pytorch
from env_model import ESM_predictor
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Union
import sys
import datetime
import esm

data_dir = 'code/evoplay/data/PAB1_GFP_data/PAB1.txt'

pab1_wt_sequence = (
    "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP"
)
starts = {
    "start_seq":
        "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALKGMLLNGQEIYFAP"  # noqa: E501
}
AAS = "ILVAGMFYWEDQNHCRKSTP"


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


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
        return ''.join(
            [AAS[i] for i in torch.argmax(states, dim=0).cpu().numpy()]
        )

    def forward(self, state_input):
        data = [
            ("seq", self._states_to_seqs(state_input_one))
            for state_input_one in state_input
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
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


def raw_to_features(data_dir, part=1):
    fr = open(data_dir, "r")
    ll = fr.readlines()
    seq_list = []
    label_list = []
    for i in range(1, len(ll)):
        tmp_list = ll[i].strip().split("\t")
        seq_list.append(tmp_list[0])
        label_list.append(float(tmp_list[1]))

    indices = random.sample(range(len(seq_list)), int(len(seq_list) / 10))
    seq_list = [seq_list[i] for i in indices]
    label_list = [label_list[i] for i in indices]
    seq_np = np.array([string_to_one_hot(seq, AAS) for seq in seq_list])

    labels = torch.from_numpy(np.array(label_list))
    labels = labels.to(torch.float32)
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots, labels


def train_cnn_predictor(data_dir, feature_extractor, part=1):
    print("training score predictor")
    one_hots, labels = raw_to_features(data_dir, part)
    seq_dataset = MyDataset(one_hots, labels)
    epochs = 1
    train_loader = DataLoader(seq_dataset, batch_size=128, shuffle=True)

    model = ESM_predictor(
        len(pab1_wt_sequence),
        len(AAS),
    ).cuda()
    optimizer = optim.AdamW(
        [params for params in model.parameters() if params.requires_grad],
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            inputs = batch[0]
            inputs = inputs.permute(0, 2, 1).cuda()
            labels = batch[1].cuda()
            optimizer.zero_grad()

            features = feature_extractor(inputs)
            features = features.mean(1)
            logits = model(features)
            loss = F.mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}, train_loss {loss}")
    return model


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
        self.n_playout = 400  # num of simulations for each move 400 1600
        self.c_puct = 10  #0.5  # 10
        self.buffer_size = 10000
        self.batch_size = 32  # mini-batch size for training  512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 50
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
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs *
                    (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
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
            1 - np.var(np.array(winner_batch) - old_v.flatten()) /
            np.var(np.array(winner_batch))
        )
        explained_var_new = (
            1 - np.var(np.array(winner_batch) - new_v.flatten()) /
            np.var(np.array(winner_batch))
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
                print()
                if self.retrain_flag and self.part <= 10:
                    print('train predictor again')

                    update_model = train_cnn_predictor(data_dir, self.part)
                    self.seq_env.model = update_model
                    self.seq_env.model.eval()
                    self.part = self.part + 1
                    self.retrain_flag = False

                    print('train predictor again done')
                    print()
                if len(
                    self.data_buffer
                ) > self.batch_size and self.buffer_no_extend == False:
                    loss, entropy = self.policy_update()
            m_p_fitness = np.array(list(self.m_p_dict.values()))
            m_p_seqs = np.array(list(self.m_p_dict.keys()))
            df_m_p = pd.DataFrame(
                {
                    "sequence": m_p_seqs, "pred_fit": m_p_fitness
                }
            )
            pathlib.Path('out/PAB1_GFP_task/generate'
                        ).mkdir(parents=True, exist_ok=True)
            df_m_p.to_csv(
                "out/PAB1_GFP_task/generate/evoplay_pab1_generated_sequence_esm_3.csv",
                index=False
            )
            endtime = datetime.datetime.now()
            print('time cost:', (endtime - starttime).seconds)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    feature_extractor = FeatureExtracter().eval().cuda()
    model = train_cnn_predictor(data_dir, feature_extractor)
    torch.cuda.empty_cache()
    training_pipeline = TrainPipeline(
        starts["start_seq"],
        AAS,
        model,
        feature_extractor,
        trust_radius=100,
    )
    training_pipeline.run()
