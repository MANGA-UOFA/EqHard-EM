import argparse
import torch
import datetime
import pathlib
import sys
import subprocess
import numpy as np
import util
import generate
import default_dataset
import evaluate_generations
import transformers
import pandas as pd

from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from dateutil import tz
from evaluate import eval_model
from pathlib import PosixPath


class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        self.encoding = 'UTF-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush

class BaseTrainer():

    LOG_ROOT_DIR = 'log/'

    def __init__(self,
        model=None,
        tokenizer=None,
        train_dataset=None,
        val_inputs=None,
        val_references=None,
        test_inputs=None,
        test_references=None,
        metrics=None,
        num_epochs=1000,
        learning_rate=5e-5,
        log_every=100,
        batch_size=64,
        gen_batch_size=64,
        save_models=True,
        log_root_dir=None,
        log_name=None,
        sanity=False,
        step_save_every=None,
        step_eval_every=None,
        epoch_save_every=None,
        epoch_eval_every=None,
        resume_path='',
        fp16=False,
        shuffle=True,
        weight_decay=None,
        track_metrics=None,
        lr_scheduler=None,
        accumulation_steps=None,
    ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tzone = tz.gettz('America/Edmonton')
        self.timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

        self.model = model.cuda()
        self.tokenizer = tokenizer

        # Set up dataloaders for the datasets
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        self.num_train_batches = len(self.train_loader)

        # Set up for optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()

        if lr_scheduler == 'constant':
            self.lr_scheduler = transformers.get_scheduler(
                'constant',
                self.optimizer,
            )
        elif lr_scheduler == 'constant_with_warmup':
            self.lr_scheduler = transformers.get_scheduler(
                'constant_with_warmup',
                self.optimizer,
                num_warmup_steps=2 * self.num_train_batches,
            )
        elif lr_scheduler == 'cosine_with_restarts':
            self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=2 * self.num_train_batches,
                num_training_steps=num_epochs * self.num_train_batches,
                num_cycles=num_epochs // 4,
            )
        else:
            raise NotImplementedError('Unknown scheduler', lr_scheduler)

        self.fp16 = fp16

        self.val_inputs = val_inputs
        self.val_references = val_references
        self.test_inputs = test_inputs
        self.test_references = test_references

        self.metrics = metrics
        self.track_metrics = track_metrics if track_metrics else []

        self.best_metrics = {}  # map from metric (str) to the best value (float)
        self.best_path    = {}  # map from metric (str) to the path of ckpt (str)

        if resume_path:

            ckpt = torch.load(PosixPath(resume_path, 'last_checkpoint.pt'))

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scaler.load_state_dict(ckpt['scaler'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

            self.training_steps = ckpt['training_steps']
            self.epoch = ckpt['epoch'] + 1
            self.global_step = ckpt['global_step']

            # logging
            self.log_root_dir = PosixPath(resume_path).parent
            self.log_dir = resume_path

        else:

            self.training_steps = 0
            self.epoch = 0 if sanity else 1
            self.global_step = 0

            # Set up for logging
            if not log_root_dir:
                log_root_dir = BaseTrainer.LOG_ROOT_DIR
            self.log_root_dir = pathlib.PosixPath(log_root_dir)
            if not self.log_root_dir.exists():
                self.log_root_dir.mkdir()

            if log_name:
                self.log_dir = pathlib.PosixPath(self.log_root_dir, self.timestamp + '_' + log_name)
            else:
                self.log_dir = pathlib.PosixPath(self.log_root_dir, self.timestamp)
            self.log_dir.mkdir()

        self.log_txt_path = pathlib.PosixPath(self.log_dir, self.timestamp + '.log')
        self.logger = Logger(self.log_txt_path)
        sys.stdout = self.logger
        sys.stderr = self.logger

        self.log_every = log_every
        self.save_models = save_models
        self.sanity = sanity
        self.batch_size = batch_size

        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.writer = SummaryWriter(log_dir=self.log_dir)  # tensorboard support

        print('> Command:', ' '.join(sys.argv))
        print()

        # print current commit info
        process = subprocess.Popen(['git', 'log', '-1'], stdout=subprocess.PIPE)
        out, err = process.communicate(timeout=5)
        print(out.decode('utf-8'))

        # Saving and Evaluation
        if step_save_every:
            self.step_save_every = step_save_every
        elif epoch_save_every:
            self.step_save_every = int(epoch_save_every * self.num_train_batches)
        else:
            self.step_save_every = self.num_train_batches

        if step_eval_every:
            self.step_eval_every = step_eval_every
        elif epoch_eval_every:
            self.step_eval_every = int(epoch_eval_every * self.num_train_batches)
        else:
            self.step_eval_every = self.num_train_batches

        self.results = None
        self.gen_batch_size = gen_batch_size

    @property
    def frac_epoch(self):
        return self.global_step / self.num_train_batches

    def compute_loss(self, batch):

        outputs = self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            labels=batch['labels'].to(self.device),
        )
        return outputs.loss

    def train_step_end(self):
        pass

    def post_eval_callback(self):
        pass

    def evaluate_helper(self, inputs, references, split):

        assert not self.model.training

        # 10-beam
        output_dir_name = 'epoch_{:.2f}_gens'.format(self.frac_epoch)
        output_dir = PosixPath(self.log_dir, output_dir_name)
        output_dir.mkdir(exist_ok=True)
        output_name = 'epoch_{:.2f}.{}'.format(self.frac_epoch, split)
        dfs = generate.generate(
            model=self.model,
            tokenizer=self.tokenizer,
            inputs=inputs,
            references=references,
            output_dir=output_dir,
            output_name=output_name,
            max_length=64,
            metrics=self.metrics,
            num_beams=10,
            num_return_sequences=10,
            batch_size=self.gen_batch_size,
        )
        sum_output_path = PosixPath(output_dir, output_name + '.sum')
        evaluate_generations.summarize(dfs, inputs, references, sum_output_path)

        results = evaluate_generations.evaluate_multi(
            dfs=dfs,
            inputs=inputs,
            references=references,
            metrics=self.metrics,
        )

        # # nucleus
        # nucleus_output_name = 'nucleus_epoch_{:.2f}.{}'.format(self.frac_epoch, split)
        # nucleus_dfs = generate.generate(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     inputs=inputs,
        #     references=references,
        #     output_dir=output_dir,
        #     output_name=nucleus_output_name,
        #     max_length=64,
        #     metrics=self.metrics,
        #     num_return_sequences=10,
        #     do_sample=True,
        #     top_p=0.95,
        #     batch_size=self.gen_batch_size,
        # )
        # nucleus_sum_output_path = PosixPath(output_dir, nucleus_output_name + '.sum')
        # evaluate_generations.summarize(nucleus_dfs, inputs, references, nucleus_sum_output_path)

        # nucleus_results = evaluate_generations.evaluate_multi(
        #     dfs=nucleus_dfs,
        #     inputs=inputs,
        #     references=references,
        #     metrics=self.metrics,
        # )
        # nucleus_results = {'greedy/' + k: v for k, v in greedy_results.items()}
        # results.update(nucleus_results)

        # greedy
        greedy_output_name = 'greedy_epoch_{:.2f}.{}'.format(self.frac_epoch, split)
        greedy_dfs = generate.generate(
            model=self.model,
            tokenizer=self.tokenizer,
            inputs=inputs,
            references=references,
            output_dir=output_dir,
            output_name=greedy_output_name,
            max_length=64,
            metrics=self.metrics,
            num_beams=1,
            num_return_sequences=1,
            batch_size=self.gen_batch_size,
        )
        greedy_output_path = PosixPath(output_dir, greedy_output_name + '.sum')
        evaluate_generations.summarize(greedy_dfs, inputs, references, greedy_output_path)

        greedy_results = evaluate_generations.evaluate_single(
            df=greedy_dfs[0],
            inputs=inputs,
            references=references,
            metrics=self.metrics,
        )

        greedy_results = {'greedy/' + k: v for k, v in greedy_results.items()}
        results.update(greedy_results)

        return results

    def evaluate(self):
        # Validation results are always required
        results = self.evaluate_helper(self.val_inputs, self.val_references, 'val')
        results = {'val/' + k: v for k, v in results.items()}
        # Test results are optional
        if self.test_inputs and self.test_references:
            test_results = self.evaluate_helper(self.test_inputs, self.test_references, 'test')
            test_results = {'test/' + k: v for k, v in test_results.items()}
            results.update(test_results)
        self.post_eval_callback()
        return results

    def epoch_end(self):
        if self.save_models:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(),
                'training_steps': self.training_steps,
                'epoch': self.epoch,
                'global_step': self.global_step,
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, PosixPath(self.log_dir, 'last_checkpoint.pt'))
        return

    def save(self):

        filename = 'epoch_{:.2f}.pt'.format(self.frac_epoch)
        torch.save(self.model, pathlib.PosixPath(self.log_dir, filename))

    def batch_handler(self, batch_idx, batch):
        '''
        '''
        if self.fp16:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(batch)
        else:
            loss = self.compute_loss(batch) / self.accumulation_steps

        if loss.requires_grad:

            self.training_steps += 1
            self.writer.add_scalar('train/steps', self.training_steps, self.global_step)

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if batch_idx % self.log_every == 0:
                self.writer.add_scalar('train/loss', loss, self.global_step)
                self.writer.add_scalar('epoch', self.epoch, self.global_step)
                print('train | epoch: {:.2f} | {}/{} | loss: {:.3f}'.format(
                    self.frac_epoch, batch_idx, self.num_train_batches, loss
                ))

            self.global_step += 1

        return

    def results_handler(self, results):

        for metric, value in results.items():

            if self.save_models:
                save = False

                if metric in self.track_metrics:

                    # initialization
                    if metric not in self.best_metrics or metric not in self.best_path:
                        save = True
                    else:
                        save = value > self.best_metrics[metric]
                        # remove previous ckpt
                        if save:
                            self.best_path[metric].unlink()

                    if save:
                        self.best_metrics[metric] = value
                        save_name = 'best_{}_epoch_{:.2f}.pt'.format(metric, self.frac_epoch)
                        # / for organizing tensorboard, but can't use / for save path
                        save_name = save_name.replace('/', '_')
                        save_path = pathlib.PosixPath(self.log_dir, save_name)
                        torch.save(self.model, save_path)
                        self.best_path[metric] = save_path

        return

    def train(self):

        # Sanity check before training
        if self.sanity and self.epoch == 0:
            self.model.eval()
            results = self.evaluate()
            print('########## Start of Results for Sanity {} ##########')
            for metric, value in results.items():
                print('{}: {}'.format(metric, value))
                self.writer.add_scalar(metric, value, global_step=self.global_step)
            print('########### End of Results for Sanity {} ###########')
            with torch.no_grad():
                if self.save_models:
                    self.save()  # save a copy of the untuned model
                self.epoch_end()
                self.epoch = 1

        # Epoch 0 is reserved for before training
        print('> start of the training loop')
        for epoch in range(self.epoch, self.num_epochs + 1):

            self.epoch = epoch

            # Training
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):

                if self.global_step % self.log_every == 0:
                    self.writer.add_scalar('train/lr', self.lr_scheduler.get_last_lr()[0], global_step=self.global_step)

                self.batch_handler(batch_idx, batch)

                self.lr_scheduler.step()

                if self.save_models:
                    if self.global_step % self.step_save_every == 0:
                        self.save()

                if self.global_step % self.step_eval_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        results = self.evaluate()

                        # Create a new dataframe for the first time
                        if self.results is None:
                            columns = sorted(list(results.keys()))
                            columns = ['epoch', 'step'] + columns
                            self.results = pd.DataFrame(columns=columns)

                        row = results.copy()
                        row['epoch'] = self.frac_epoch
                        row['step'] = self.global_step
                        self.results = self.results.append(row, ignore_index=True)
                        self.results.to_csv(PosixPath(self.log_dir, 'results.csv'), index=False)

                        self.results_handler(results)

                        # logging evaluation results
                        for metric, value in results.items():
                            self.writer.add_scalar(metric, value, global_step=self.global_step)

                    self.model.train()

                self.train_step_end()

            # End of Epoch
            with torch.no_grad():
                self.epoch_end()

            print('end of epoch {}'.format(epoch))

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
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--track-metrics', nargs='*')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--resume-path', default='')

    # Model parameters
    parser.add_argument('--model-str', type=str, required=True)

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gen-batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--lr-scheduler', type=str, default='constant', choices=['constant', 'constant_with_warmup', 'cosine_with_restarts'])
    parser.add_argument('--accumulation-steps', type=int, default=1)

    # Evaluation parameters
    parser.add_argument('--language', type=str, choices=['en', 'zh'], default='en')
    parser.add_argument('--multi-ref', action='store_true')
    parser.add_argument('--smooth-bleus', action='store_true')
    parser.add_argument('--rebuild-dataset', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = util.get_tokenizer(args.model_str)

    model = AutoModelWithLMHead.from_pretrained(args.model_str)
    model.resize_token_embeddings(len(tokenizer))

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

    dataset = default_dataset.DefaultDataset(
        tokenizer,
        max_length=64,
        path=args.train_path,
        rebuild=args.rebuild_dataset,
        lower=True if args.language == 'en' else False
    )

    if args.num_training_examples:
        indices = np.random.choice(len(dataset), size=args.num_training_examples, replace=False)
        train_dataset = torch.utils.data.Subset(dataset, indices=indices)
    else:
        train_dataset = dataset

    metrics = evaluate_generations.Metrics(language=args.language, smooth=args.smooth_bleus)

    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_inputs=val_inputs,
        val_references=val_references,
        test_inputs=test_inputs,
        test_references=test_references,
        metrics=metrics,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        step_save_every=args.step_save_every,
        step_eval_every=args.step_eval_every,
        epoch_save_every=args.epoch_save_every,
        epoch_eval_every=args.epoch_eval_every,
        batch_size=args.batch_size,
        gen_batch_size=args.gen_batch_size,
        save_models=not args.no_save,
        log_root_dir=args.log_root_dir,
        log_name=args.log_name,
        sanity=args.sanity,
        resume_path=args.resume_path,
        fp16=args.fp16,
        weight_decay=args.weight_decay,
        track_metrics=args.track_metrics,
        lr_scheduler=args.lr_scheduler,
        accumulation_steps=args.accumulation_steps,
    )

    trainer.train()
