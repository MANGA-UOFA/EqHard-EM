import argparse
import torch
import datetime
import pathlib
import sys
import subprocess
import json
import numpy as np
import util
import esnli
import generate
import default_dataset
import evaluate_generations
import math
import pandas as pd

from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dateutil import tz
from evaluate import eval_model
from pathlib import PosixPath
from base_trainer import BaseTrainer
from torch.nn.functional import cross_entropy
from scipy.optimize import linear_sum_assignment
from model import adapter_t5
from abc import ABC, abstractmethod
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Base trainer script')

    parser.add_argument('--num-training-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--train-path', type=str, required=True)

    parser.add_argument('--language', type=str, choices=['en', 'zh'], default='en')

    # Model parameters
    parser.add_argument('--ckpt-path', type=str, required=True)

    # Training parameters
    parser.add_argument('--trainer', choices=['eqhem', 'random', 'sem', 'hem', 'drandom'], required=True)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=640)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # path
    if args.ckpt_path:
        model = torch.load(args.ckpt_path).cuda()
        model.eval()
        if args.language == 'en':
            tokenizer = util.get_tokenizer('t5-small')
        else:
            tokenizer = util.get_tokenizer('uer/t5-small-chinese-cluecorpussmall')

    # dataset
    dataset = default_dataset.DefaultDataset(tokenizer, max_length=64, path=args.train_path, rebuild=False)
    if args.num_training_examples:
        indices = np.random.choice(len(dataset), size=args.num_training_examples, replace=False)
        train_dataset = torch.utils.data.Subset(dataset, indices=indices)
    else:
        train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    tzone = tz.gettz('America/Edmonton')
    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    # df = pd.DataFrame(columns=['Decoder {}'.format(i) for i in range(model.num_modes)])
    df = pd.DataFrame()

    with torch.no_grad():

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            # ----- Get per_sample_loss -----
            labels = batch['labels'].cuda()
            batch_size = len(labels)
            num_modes = model.num_modes
            per_sample_loss = torch.zeros(batch_size, num_modes)
            encoder_outputs = model.encode(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
            )
            for i in range(num_modes):
                model.mode_idx = i
                outputs = model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=batch['attention_mask'].cuda(),
                    labels=labels,
                )
                logits = outputs.logits
                for sample_idx in range(batch_size):
                    per_sample_loss[sample_idx][i] = cross_entropy(logits[sample_idx], labels[sample_idx])
                del outputs

            # ----- Get assignment -----
            if args.trainer == 'hem':
                assignments = torch.zeros(batch_size, num_modes)
                mode_assignments = per_sample_loss.argmin(dim=-1)
                assignments[range(batch_size), mode_assignments] = 1  # (batch_size, num_modes): each entry is 0 or 1

            elif args.trainer == 'sem':
                assignments = torch.softmax(-per_sample_loss, dim=-1)  # (batch_size, num_decoder)

            elif args.trainer == 'eqhem':

                batch_size = len(per_sample_loss)
                num_modes = model.num_modes

                assignments = torch.zeros(batch_size, num_modes)

                samples_per_mode = math.ceil(batch_size / num_modes)
                cost = per_sample_loss.detach().cpu().numpy().repeat(samples_per_mode, axis=1)

                _, mode_assignments = linear_sum_assignment(cost)

                mode_assignments //= samples_per_mode  # (batch_size,): each entry is [0, num_modes]
                assignments[range(batch_size), mode_assignments] = 1  # (batch_size, num_modes): each entry is 0 or 1

            # ----- log assignment -----
            df = df.append(assignments.tolist())

    df.to_csv('{}_'.format(args.trainer) + timestamp + '.assignment_stats', index=False)
