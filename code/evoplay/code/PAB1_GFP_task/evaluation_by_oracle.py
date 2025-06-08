import pathlib
import tape
import torch
import numpy as np
import pandas as pd
# import datetime
device = "cuda:0"

tokenizer = tape.TAPETokenizer(vocab="iupac")

model = tape.ProteinBertForValuePrediction.from_pretrained(
    "code/evoplay/data/Oracle_weight/landscape_params/tape_landscape/Pab1"
).to(device)

seqs = []
all_score = []
gen_path = pathlib.Path('out/PAB1_GFP_task/PAB1/generate')
eval_path = pathlib.Path('out/PAB1_GFP_task/PAB1/evaluate')
file_name = 'evoplay_pab1_generated_sequence.csv'

if not (gen_path / file_name).exists():
    print("file not exist")
    exit(0)
eval_path.mkdir(parents=True, exist_ok=True)

df_2 = pd.read_csv(gen_path / file_name)
sequences = list(df_2['sequence'])
print('sequence counts', len(sequences))
score_list = []
for sequence in sequences:
    encoded_seqs = torch.tensor(tokenizer.encode(sequence)
                               ).unsqueeze(0).to(device)
    score = model(encoded_seqs)[0].detach().cpu().numpy().astype(float
                                                                ).reshape(-1)
    score_list.append(score[0])

print("predict score length", len(score_list))
evalute_df = pd.DataFrame({'sequence': sequences, 'score': score_list})
evalute_df.sort_values("score", inplace=True, ascending=False)
evalute_df.to_csv(eval_path / file_name, index=False)
print(evalute_df.head(10))
