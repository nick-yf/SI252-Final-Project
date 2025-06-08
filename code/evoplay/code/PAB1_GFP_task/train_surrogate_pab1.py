import pathlib

import esm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from env_model import CNN2, ESM_predictor
from residue_constant import AAS

data_path = 'code/evoplay/data/PAB1_GFP_data/PAB1.txt'
pab1_wt_sequence = (
    "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP"
)


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


def train_surrogate_cnn_predictor(data_dir):
    print("training cnn score predictor")
    one_hots, labels = raw_to_features(data_dir)
    seq_dataset = MyDataset(one_hots, labels)
    epochs = 5
    train_loader = DataLoader(seq_dataset, batch_size=128, shuffle=True)

    model = CNN2(
        len(pab1_wt_sequence),
        len(AAS),
    ).cuda()
    optimizer = optim.Adam(
        [params for params in model.parameters() if params.requires_grad],
    )

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            inputs = batch[0].cuda()
            labels = batch[1].cuda()
            optimizer.zero_grad()

            logits = model(inputs.permute(0, 2, 1)).squeeze()
            loss = F.mse_loss(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}, train_loss {loss}")
    pathlib.Path('out/PAB1_GFP_task/PAB1').mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        "out/PAB1_GFP_task/PAB1/surrogate_predictor_cnn2.pth"
    )
    print("Surrogate predictor trained and saved.")
    torch.cuda.empty_cache()


def train_surrogate_esm_predictor(data_dir, feature_extractor):
    print("training esm score predictor")
    one_hots, labels = raw_to_features(data_dir)
    seq_dataset = MyDataset(one_hots, labels)
    epochs = 10
    train_loader = DataLoader(seq_dataset, batch_size=512, shuffle=True)

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
    pathlib.Path('out/PAB1_GFP_task/PAB1').mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        "out/PAB1_GFP_task/PAB1/surrogate_predictor_esm.pth"
    )
    print("Surrogate predictor trained and saved.")


class MyDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, index):
        seq, target = self.sequences[index], self.labels[index]
        return seq, target

    def __len__(self):
        return len(self.sequences)


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


if __name__ == '__main__':
    train_surrogate_cnn_predictor(data_path)
    # feature_extractor = FeatureExtracter().eval().cuda()
    # train_surrogate_esm_predictor(data_path, feature_extractor)
