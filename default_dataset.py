import torch
import pandas as pd
import generate
import evaluate_generations
import util
import time
import pickle

from pathlib import PosixPath
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead
from collections import OrderedDict
from tqdm import tqdm


def _squeeze_helper(x):
    '''Helper function to squeeze each value of the given dictionary
    '''
    return {k: v.squeeze() for k, v in x.items()}


class DefaultDataset(Dataset):

    def __init__(self, tokenizer, max_length=64, path=None, rebuild=False, lower=False):


        cache_file = path.replace('.csv', '.cache')

        if not PosixPath(cache_file).exists() or rebuild:

            print('Building dataset...')

            df = pd.read_csv(path)

            if lower:
                df['context'] = df['context'].apply(lambda x: x.lower())
                df['response'] = df['response'].apply(lambda x: x.lower())

            if 'gpt2' in tokenizer.name_or_path or 'DialoGPT' in tokenizer.name_or_path:

                raise NotImplementedError()

                # contexts = []
                # responses = []

                # samples = []

                # for index, row in df.iterrows():
                #     context = row['context'].strip().lower()
                #     response = row['response'].strip().lower()
                #     sample = '{} {} {} {}'.format(context, tokenizer.sep_token, response, tokenizer.eos_token)
                #     samples.append(sample)

                # encoded = tokenizer(
                #     samples,
                #     max_length=max_length * 2,
                #     truncation=True,
                #     padding='max_length',
                #     return_tensors='pt',
                # )

                # self.data = dict()
                # self.data['input_ids'] = encoded['input_ids']
                # self.data['attention_mask'] = encoded['attention_mask']
                # self.data['labels'] = encoded['input_ids'].clone()
                # self.data['labels'][encoded['attention_mask'] == 0] = -100

            else:

                self.data = dict()
                self.data['input_ids'] = torch.zeros((len(df), max_length), dtype=torch.long)
                self.data['attention_mask'] = torch.zeros((len(df), max_length), dtype=torch.long)
                self.data['labels'] = torch.zeros((len(df), max_length), dtype=torch.long)

                for idx in tqdm(range(len(df)), mininterval=10):

                    input = tokenizer(
                        df['context'][idx] + ' ' + tokenizer.eos_token,
                        max_length=max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                        add_special_tokens=False,
                    )
                    label = tokenizer(
                        df['response'][idx] + ' ' + tokenizer.eos_token,
                        max_length=max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                        add_special_tokens=False,
                    )
                    label['input_ids'][label['attention_mask'] == 0] = -100

                    input = _squeeze_helper(input)
                    label = _squeeze_helper(label)

                    self.data['input_ids'][idx] = input['input_ids']
                    self.data['attention_mask'][idx] = input['attention_mask']
                    self.data['labels'][idx] = label['input_ids']

                with open(cache_file, mode='wb') as f:
                    pickle.dump(
                        {
                            'input_ids': self.data['input_ids'],
                            'attention_mask': self.data['attention_mask'],
                            'labels': self.data['labels']
                        }, f
                    )

        else:
            print('Loading from cache...')

            with open(cache_file, mode='rb') as f:
                data = pickle.load(f)
            self.data = data

    def __getitem__(self, index):
        return {
            'input_ids': self.data['input_ids'][index],
            'attention_mask': self.data['attention_mask'][index],
            'labels': self.data['labels'][index],
            'indices': index,
        }

    def __len__(self):
        return len(self.data['input_ids'])

def get_inputs_and_references(path, multi_ref=False):

    df = pd.read_csv(path)
    total = len(df)
    df = df.dropna()
    if total != len(df):
        print('[Warning]: {} samples have been dropped due to NaNs, file: {}'.format(total - len(df), path))

    if multi_ref:
        tests = OrderedDict()
        for idx, row in df.iterrows():
            input = row['context'].lower()
            reference = row['response'].lower()
            if input in tests:
                tests[input].append(reference)
            else:
                tests[input] = [reference]
        inputs = list(tests.keys())
        references = list(tests.values())
    else:
        inputs =  list(df['context'].apply(lambda x: x.lower()))
        references = list(map(lambda x: [x.lower()], list(df['response'])))

    return inputs, references

if __name__ == '__main__':

    # tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # model = AutoModelWithLMHead.from_pretrained("t5-base")
    # dataset = DefaultDataset(tokenizer)

    # model = torch.load('ckpts/dd/best_dd_ckpt.pt').cuda()
    # tokenizer = util.get_tokenizer('t5-small')

    # inputs, references = get_inputs_and_references('data/cleaned_dd/single-turn/test.csv')

    # start_time = time.time()
    # df = generate.generate(
    #     model=model,
    #     tokenizer=tokenizer,
    #     inputs=inputs,
    #     references=references,
    #     output_path='sample_generate.csv',
    #     max_length=64,
    # )
    # print('took:', time.time() - start_time)

    # results = evaluate_generations.evaluate_single(df, inputs, references)
    # print(results)

    tokenizer = util.get_tokenizer('t5-small')
    data_path = 'data/ost_tiny/ost_tiny.csv'

    inputs, references = get_inputs_and_references(data_path)
    dataset = DefaultDataset(tokenizer, path=data_path, rebuild=True)
