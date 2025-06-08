from tqdm import tqdm

import esm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from train_m_single_m_p_pab1_esm import string_to_one_hot, raw_to_features, MyDataset, FeatureExtracter
from residue_constant import AAS

extracter = FeatureExtracter().eval().cuda()

pab1_data = "code/evoplay/data/PAB1_GFP_data/PAB1.txt"
gfp_data = "code/evoplay/data/PAB1_GFP_data/GFP_237.txt"

# one_hots_pab1, labels_pab1 = raw_to_features(pab1_data)
# dataset = MyDataset(one_hots_pab1, labels_pab1)
# train_loader = DataLoader(
#     dataset, batch_size=128, shuffle=False, drop_last=False
# )

# all_features = []
# all_labels = []
# for i in tqdm(train_loader):
#     inputs = i[0].cuda()
#     labels = i[1]

#     features = extracter(inputs)
#     features = features.mean(1)
#     features = features.cpu().detach()

#     all_features.append(features)
#     all_labels.append(labels)
# all_features = torch.cat(all_features, dim=0)
# all_labels = torch.cat(all_labels, dim=0)
# torch.save(
#     {
#         "features": all_features,
#         "labels": all_labels,
#     },
#     "code/evoplay/data/PAB1_GFP_data/pab1_features.pt",
# )

print("Visualizing PAB1 features...")
generated_sequences = [
    "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALKGMLLNGQNIYVAP",
    "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEMAAKEAIDALKGMLLNGQEIYVAP"
]
inputs = torch.stack(
    [string_to_one_hot(seq, AAS) for seq in generated_sequences]
).cuda()
features = extracter(inputs)
features = features.mean(1)
generated_features = features.cpu().detach().numpy()

data = torch.load("code/evoplay/data/PAB1_GFP_data/pab1_features.pt")
all_features = data["features"]
all_labels = data["labels"]

ordered_indices = torch.argsort(all_labels)
all_features = all_features[ordered_indices]
all_labels = all_labels[ordered_indices]

features_np = all_features.numpy()
labels_np = all_labels.numpy()

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
all_features_2d = tsne.fit_transform(
    np.concatenate([features_np, generated_features], axis=0)
)
generated_features_2d = all_features_2d[-2:, :]
features_2d = all_features_2d[:-2, :]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    c=labels_np,
    cmap='viridis',
    alpha=1.0,
    s=10,
    edgecolor='none',
    label='Experimental Validated PAB1 Sequences',
)
ax.scatter(
    generated_features_2d[0, 0],
    generated_features_2d[0, 1],
    marker='*',
    c='red',
    edgecolors='black',
    linewidths=1.0,
    s=100,
    label='Generated Sequence By EvoPlay with ESM2',
)
ax.scatter(
    generated_features_2d[1, 0],
    generated_features_2d[1, 1],
    marker='*',
    c='white',
    edgecolors='black',
    linewidths=1.0,
    s=100,
    label='Generated Sequence By Original EvoPlay',
)
ax.legend()
cbar = fig.colorbar(scatter, ax=ax, label='Functional Score')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig('pab1_features_tsne.png', dpi=600)
print("PAB1 features visualization saved as pab1_features_tsne.png")

# one_hots, labels = raw_to_features(gfp_data)
# dataset = MyDataset(one_hots, labels)
# train_loader = DataLoader(
#     dataset, batch_size=128, shuffle=False, drop_last=False
# )

# all_features = []
# all_labels = []
# for i in tqdm(train_loader):
#     inputs = i[0].cuda()
#     labels = i[1]

#     features = extracter(inputs)
#     features = features.mean(1)
#     features = features.cpu().detach()

#     all_features.append(features)
#     all_labels.append(labels)
# all_features = torch.cat(all_features, dim=0)
# all_labels = torch.cat(all_labels, dim=0)
# torch.save(
#     {
#         "features": all_features,
#         "labels": all_labels,
#     },
#     "code/evoplay/data/PAB1_GFP_data/gfp_features.pt",
# )

# print("Visualizing GFP features...")
# data = torch.load("code/evoplay/data/PAB1_GFP_data/gfp_features.pt")
# all_features = data["features"]
# all_labels = data["labels"]

# ordered_indices = torch.argsort(all_labels)
# all_features = all_features[ordered_indices]
# all_labels = all_labels[ordered_indices]

# features_np = all_features.numpy()
# labels_np = all_labels.numpy()**2

# # Reduce to 2D using t-SNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=50)
# features_2d = tsne.fit_transform(features_np)

# # Plot
# fig, ax = plt.subplots(figsize=(8, 6))
# scatter = ax.scatter(
#     features_2d[:, 0],
#     features_2d[:, 1],
#     c=labels_np,
#     cmap='viridis',
#     alpha=1.0,
#     s=10,
#     edgecolor='none',
# )
# cbar = fig.colorbar(scatter, ax=ax)
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.tight_layout()
# plt.savefig('gfp_features_tsne.png', dpi=600)
# print("GFP features visualization saved as gfp_features_tsne.png")
