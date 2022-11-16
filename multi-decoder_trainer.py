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
from model import multidecoder_t5
from abc import ABC, abstractmethod
from torch import nn


class MultiAdapterTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_trained_per_mode = [0] * self.model.num_modes

    def evaluate_helper(self, inputs, references, split):

        assert not self.model.training

        dfs = []

        output_dir_name = 'epoch_{:.2f}_gens'.format(self.frac_epoch)
        output_dir = PosixPath(self.log_dir, output_dir_name)
        output_dir.mkdir(exist_ok=True)

        for mode_idx in range(self.model.num_modes):

            output_name = 'epoch_{:.2f}.mode_{}.{}'.format(self.frac_epoch, mode_idx, split)

            self.model.mode_idx = mode_idx
            df = generate.generate(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                references=references,
                max_length=64,
                metrics=self.metrics,
                output_dir=output_dir,
                output_name=output_name,
                num_beams=1,
                num_return_sequences=1,
            )
            assert len(df) == 1
            dfs.append(df[0])

        sum_output_path = PosixPath(output_dir, output_name + '.sum')
        evaluate_generations.summarize(dfs, inputs, references, sum_output_path)

        single_results = evaluate_generations.evaluate_single_multi(dfs, inputs, references, self.metrics)
        multi_results  = evaluate_generations.evaluate_multi(dfs, inputs, references, self.metrics)

        results = {'single/' + k: v for k, v in single_results.items()}
        results.update({'multi/' + k: v for k, v in multi_results.items()})

        return results

    def post_eval_callback(self):
        total_steps = sum(self.steps_trained_per_mode)
        if total_steps > 0:
            print('> Percent trained (epoch {:.2f}):'.format(self.frac_epoch))
            for i in range(len(self.steps_trained_per_mode)):
                print('Mode [{}]: {:.2f} %'.format(i, self.steps_trained_per_mode[i] / total_steps * 100))
        return

class EqualSizeHardEmTrainer(MultiAdapterTrainer):

    def __init__(self,
            *args,
            num_forward_passes=1,
            num_steps_per_assignment=1,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.ongoing_batch = None
        self.per_sample_loss = None

        self.num_forward_passes = num_forward_passes
        self.num_steps_per_assignment = num_steps_per_assignment

        # the batch size for each decoder update
        self.effective_bz = self.batch_size * self.num_forward_passes // self.model.num_modes

    def batch_handler(self, batch_idx, batch):

        per_sample_loss = self.get_per_sample_loss(batch)

        if self.ongoing_batch is None and self.per_sample_loss is None:
            self.ongoing_batch = batch
            self.per_sample_loss = per_sample_loss
        else:
            # concatenate the batch
            for k, v in batch.items():
                self.ongoing_batch[k] = torch.concat((self.ongoing_batch[k], v), 0)
            # concatenate the per_sample_loss
            self.per_sample_loss = torch.concat((self.per_sample_loss, per_sample_loss), 0)

        if (batch_idx + 1) % (self.num_forward_passes * self.num_steps_per_assignment) == 0:

            assignments = self.assign(self.per_sample_loss)

            # handle each mode
            for i in range(self.model.num_modes):

                # skip if nothing is assigned to the mode
                mode_i_sample_indices = (assignments==i).nonzero()[0]
                if len(mode_i_sample_indices) < 1:
                    continue

                for j in range(self.num_steps_per_assignment):

                    # global step is 0-indexed, if num_steps is 1, then +1-1 cancels
                    global_step = self.global_step + 1 - self.num_steps_per_assignment + j
                    # effective index is 1-indexed
                    effective_batch_idx = batch_idx + 2 - self.num_steps_per_assignment + j

                    log = effective_batch_idx % self.log_every == 0

                    # prepare the batch data for the mode
                    mode_i_mini_batch = dict()
                    jth_mode_i_sample_indices = mode_i_sample_indices[j * self.effective_bz: (j + 1) * self.effective_bz]

                    for k, v in self.ongoing_batch.items():
                        mode_i_mini_batch[k] = v[jth_mode_i_sample_indices]

                    # calculate loss
                    self.model.mode_idx = i
                    loss = self.compute_loss(mode_i_mini_batch)

                    if loss.requires_grad:

                        self.training_steps += 1
                        self.writer.add_scalar('train/steps', self.training_steps, global_step)

                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    if log:
                        self.writer.add_scalars(
                            'training_loss',
                            {
                                'a_{}'.format(i): loss
                            },
                            global_step,
                        )
                        print('train | mode: {} | epoch: {:.2f} | {}/{} | loss: {:.3f}'.format(
                            i, self.frac_epoch, effective_batch_idx, self.num_train_batches, loss
                        ))

            self.per_sample_loss = None
            self.ongoing_batch = None

        self.global_step += 1

        return

    def compute_loss(self, batch):
        self.steps_trained_per_mode[self.model.mode_idx] += 1
        return super().compute_loss(batch)

    def get_per_sample_loss(self, batch):

        with torch.no_grad():

            # TODO: process this in the Dataloader
            labels = batch['labels'].cuda()
            batch_size = len(labels)
            num_modes = self.model.num_modes

            per_sample_loss = torch.zeros(batch_size, num_modes)

            encoder_outputs = self.model.encode(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
            )

            for i in range(num_modes):

                self.model.mode_idx = i
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=batch['attention_mask'].cuda(),
                    labels=labels,
                )
                logits = outputs.logits

                for sample_idx in range(batch_size):
                    per_sample_loss[sample_idx][i] = cross_entropy(logits[sample_idx], labels[sample_idx])

                del outputs

        return per_sample_loss

    def assign(self, per_sample_loss):

        batch_size = len(per_sample_loss)
        num_modes = self.model.num_modes

        samples_per_mode = math.ceil(batch_size / num_modes)
        cost = per_sample_loss.detach().cpu().numpy().repeat(samples_per_mode, axis=1)

        _, mode_assignments = linear_sum_assignment(cost)
        mode_assignments //= samples_per_mode

        return mode_assignments

class RandomTrainer(MultiAdapterTrainer):

    def compute_loss(self, batch):
        self.steps_trained_per_mode[self.model.mode_idx] += 1
        return super().compute_loss(batch)

    def epoch_begin(self):
        self.model.mode_idx = 0

    def train_step_end(self):
        self.model.mode_idx = (self.model.mode_idx + 1) % self.model.num_modes

class EMTrainer(MultiAdapterTrainer, ABC):

    @abstractmethod
    def handle_per_sample_loss(self, per_sample_loss):
        '''Given a matrix (num_samples x num_modes), return the loss and the
        number of steps trained on each mode
        '''
        pass

    def compute_loss(self, batch):

        num_modes = self.model.num_modes

        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        attention_mask = batch['attention_mask'].cuda()

        batch_size = len(labels)
        per_sample_loss = torch.zeros(batch_size, num_modes).cuda()

        encoder_outputs = self.model.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        for i in range(num_modes):

            self.model.mode_idx = i
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits

            for sample_idx in range(batch_size):
                per_sample_loss[sample_idx][i] = cross_entropy(logits[sample_idx], labels[sample_idx])

        loss, per_decoder_steps = self.handle_per_sample_loss(per_sample_loss)
        if self.model.training:
            for j in range(num_modes):
                self.steps_trained_per_mode[j] += per_decoder_steps[j].item()

        return loss

class DropoutTrickEMTrainer(MultiAdapterTrainer, ABC):

    def __init__(self, *args, lp=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.lp = lp
        if self.lp:
            d_model = self.model.config.d_model
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(d_model // 4, self.model.num_modes),
                nn.Softmax(),
            ).cuda()
            self.optimizer.add_param_group({'params': [p for p in self.classifier.parameters()]})

    def get_per_sample_loss(self, batch):
        '''
        Args:
            consider_prior: if True, no longer calculates the per_sample_loss.
            Rather, modifiy the per_sample_loss so that the get_assignments call
            will return the learned_prior modified assignments.
        '''

        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        attention_mask = batch['attention_mask'].cuda()

        batch_size = len(labels)

        encoder_outputs = self.model.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        per_sample_loss = torch.zeros(batch_size, self.model.num_modes).cuda()

        for i in range(self.model.num_modes):

            self.model.mode_idx = i
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits

            for sample_idx in range(batch_size):
                per_sample_loss[sample_idx][i] = cross_entropy(logits[sample_idx], labels[sample_idx])

        if self.lp:
            assert self.lp == True
            sums = (encoder_outputs.last_hidden_state * attention_mask.unsqueeze(dim=-1)).sum(dim=1)
            embeddings = sums / attention_mask.sum(dim=1).unsqueeze(dim=-1)

            priors = self.classifier(embeddings)
            likelihood = torch.exp(-per_sample_loss)
            posterior = priors * likelihood
            per_sample_loss = -torch.log(posterior)

        return per_sample_loss

    @abstractmethod
    def get_assignments(self, per_sample_loss):
        '''Given per_sample_loss (num_samples x num_decoders), return a matrix
        of the same size, indicating the weight of each sample forf each decoder
        '''
        pass

    def compute_loss(self, batch):

        num_modes = self.model.num_modes
        batch_size = len(batch['labels'])

        if self.model.training:
            with torch.no_grad():
                self.model.eval()
                no_dropout_per_sample_loss = self.get_per_sample_loss(batch)
                self.model.train()
                assignments = self.get_assignments(no_dropout_per_sample_loss)

        per_sample_loss = self.get_per_sample_loss(batch)
        loss = (per_sample_loss * assignments).sum() / batch_size

        with torch.no_grad():
            per_decoder_steps = torch.sum(assignments, dim=0)
            per_decoder_steps = per_decoder_steps.cpu().numpy() / batch_size

        if self.model.training:
            for j in range(num_modes):
                self.steps_trained_per_mode[j] += per_decoder_steps[j].item()

        return loss

class HardEMTrainer(EMTrainer):

    def handle_per_sample_loss(self, per_sample_loss):

        num_modes = self.model.num_modes
        batch_size = per_sample_loss.shape[0]

        losses, indices = torch.min(per_sample_loss, dim=-1)
        loss = losses.sum() / batch_size

        per_decoder_steps = torch.histc(
            indices, min=0, max=num_modes-1, bins=num_modes
        ).cpu().numpy()
        per_decoder_steps = per_decoder_steps / batch_size

        return loss, per_decoder_steps

class SoftEMTrainer(EMTrainer):

    def handle_per_sample_loss(self, per_sample_loss):

        batch_size = per_sample_loss.shape[0]

        with torch.no_grad():
            per_sample_per_decoder_prob = torch.softmax(-per_sample_loss, dim=-1)  # (batch_size, num_decoder)
            per_decoder_steps = torch.sum(per_sample_per_decoder_prob, dim=0)
        per_sample_loss *= per_sample_per_decoder_prob  # weigh each

        loss = torch.sum(per_sample_loss) / batch_size
        per_decoder_steps = per_decoder_steps.cpu().numpy() / batch_size

        return loss, per_decoder_steps

class DropoutTrickHardEMTrainer(DropoutTrickEMTrainer):

    def get_assignments(self, per_sample_loss):

        num_modes = self.model.num_modes
        batch_size = per_sample_loss.shape[0]

        assignments = torch.zeros((batch_size, num_modes))
        best_decoders = per_sample_loss.argmin(dim=1)
        assignments[range(batch_size), best_decoders] = 1

        return assignments.cuda()

class DropoutTrickSoftEMTrainer(DropoutTrickEMTrainer):

    def get_assignments(self, per_sample_loss):

        num_modes = self.model.num_modes
        batch_size = per_sample_loss.shape[0]

        batch_size = per_sample_loss.shape[0]

        assignments = torch.softmax(-per_sample_loss, dim=-1)  # (batch_size, num_decoder)

        return assignments.cuda()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Base trainer script')

    parser.add_argument('--num-training-examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--step-save-every', type=int, default=None)
    parser.add_argument('--step-eval-every', type=int, default=None)
    parser.add_argument('--epoch-save-every', type=float, default=None)
    parser.add_argument('--epoch-eval-every', type=float, default=None)

    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--val-path',  type=str, required=True)
    parser.add_argument('--test-path',  type=str, required=False)

    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--log-root-dir', type=str, default=BaseTrainer.LOG_ROOT_DIR)
    parser.add_argument('--log-name', type=str, default='')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--track-metrics', nargs='*')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--resume-path', default='')

    # Model parameters
    parser.add_argument('--model-str', type=str, required=True)
    parser.add_argument('--num-modes', type=int, default=10)
    parser.add_argument('--init-ckpt', type=str, required=True)
    parser.add_argument('--freeze', action='store_true', help='if enabled, only finetune adapters')
    parser.add_argument('--proj-ratio', type=float, default=0.5)

    # Training parameters
    parser.add_argument('--trainer', choices=['eqhem', 'random', 'sem', 'hem', 'drandom', 'trick-hem', 'trick-sem'], required=True)
    parser.add_argument('--decoder', choices=['adapter', 'transformer'], default='adapter')
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gen-batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--lr-scheduler', type=str, default='constant', choices=['constant', 'constant_with_warmup', 'cosine_with_restarts'])
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--num-forward-passes', type=int, default=1)  # specific to eqhem trainer
    parser.add_argument('--num-steps-per-assignment', type=int, default=1)  # specific to eqhem trainer
    parser.add_argument('--lp', action='store_true')

    # Evaluation parameters
    parser.add_argument('--language', type=str, choices=['en', 'zh'], default='en')
    parser.add_argument('--multi-ref', action='store_true')
    parser.add_argument('--smooth-bleus', action='store_true')
    parser.add_argument('--rebuild-dataset', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = util.get_tokenizer(args.model_str)

    # loading the model
    init_ckpt = torch.load(args.init_ckpt)
    config = init_ckpt.config
    config.adapter_hidden_size = int(config.d_model * args.proj_ratio)
    if args.decoder == 'adapter':
        model = adapter_t5.MultiAdapterT5(config, num_modes=args.num_modes)
    elif args.decoder == 'transformer':
        model = multidecoder_t5.MultiDecoderT5(config, num_modes=args.num_modes)
    else:
        raise NotImplementedError()
    model.from_single_pretrained(init_ckpt.state_dict())
    model.train()
    del init_ckpt
    model.resize_token_embeddings(len(tokenizer))

    if args.decoder == 'adapter':
        if args.freeze:
            for k, v in model.named_parameters():
                v.requires_grad = False
                if 'adapter' in k:
                    v.requires_grad = True

    if args.val_path:
        val_inputs, val_references = default_dataset.get_inputs_and_references(
            args.val_path,
            multi_ref=args.multi_ref,
        )
    if args.test_path:
        test_inputs, test_references = default_dataset.get_inputs_and_references(
            args.test_path,
            multi_ref=args.multi_ref,
        )
    else:
        test_inputs = None
        test_references = None

    dataset = default_dataset.DefaultDataset(tokenizer, max_length=64, path=args.train_path, rebuild=args.rebuild_dataset)

    if args.num_training_examples:
        indices = np.random.choice(len(dataset), size=args.num_training_examples, replace=False)
        train_dataset = torch.utils.data.Subset(dataset, indices=indices)
    else:
        train_dataset = dataset

    metrics = evaluate_generations.Metrics(language=args.language, smooth=args.smooth_bleus)

    config = {
        'model': model,
        'tokenizer': tokenizer,
        'train_dataset': train_dataset,
        'val_inputs': val_inputs,
        'val_references': val_references,
        'test_inputs': test_inputs,
        'test_references': test_references,
        'metrics': metrics,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'log_every': args.log_every,
        'batch_size': args.batch_size,
        'gen_batch_size': args.gen_batch_size,
        'save_models': not args.no_save,
        'log_root_dir': args.log_root_dir,
        'log_name': args.log_name,
        'sanity': args.sanity,
        'resume_path': args.resume_path,
        'fp16': args.fp16,
        'weight_decay': args.weight_decay,
        'lr_scheduler': args.lr_scheduler,
        'step_save_every': args.step_save_every,
        'step_eval_every': args.step_eval_every,
        'epoch_save_every': args.epoch_save_every,
        'epoch_eval_every': args.epoch_eval_every,
        'track_metrics': args.track_metrics,
        'accumulation_steps': args.accumulation_steps,
    }

    if args.trainer == 'eqhem':
        config['num_forward_passes'] = args.num_forward_passes
        config['num_steps_per_assignment'] = args.num_steps_per_assignment
        config['batch_size'] = (args.batch_size * args.num_modes) // args.num_forward_passes
        trainer = EqualSizeHardEmTrainer(**config)
    elif args.trainer == 'hem':
        trainer = HardEMTrainer(**config)
    elif args.trainer == 'sem':
        trainer = SoftEMTrainer(**config)
    elif args.trainer == 'random':
        config['shuffle'] = False
        trainer = RandomTrainer(**config)
    elif args.trainer == 'drandom':
        config['shuffle'] = True
        trainer = RandomTrainer(**config)
    elif args.trainer == 'trick-hem':
        trainer = DropoutTrickHardEMTrainer(lp=args.lp, **config)
    elif args.trainer == 'trick-sem':
        trainer = DropoutTrickSoftEMTrainer(lp=args.lp, **config)

    trainer.train()
