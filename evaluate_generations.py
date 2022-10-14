import argparse
import util
import torch
import sys
import math
import numpy as np
import pandas as pd
import copy
import time
import datetime
import sacrebleu

import itertools
import statistics
import generate
import default_dataset
import nltk
import multiprocess as mp

from esnli import build_esnli_tests
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from pathlib import PosixPath
from typing import List
from dateutil import tz
from tqdm import tqdm
from transformers import AutoTokenizer


def time_profile(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('Function [{}] took {:.2f} seconds.'.format(f.__name__, te-ts))
        return result
    return timed

class Metrics():

    def __init__(self, language='en', pkg='nltk', smooth=False):
        '''
        Arguments:
            language [str]: en for English and zh for Chinese
            pkg [str]: nltk or sacrebleu
        '''

        self.language = language
        self.pkg = pkg
        self.smooth = smooth

        # set up tokenizing API
        if pkg == 'nltk':
            if language == 'en':
                self.tokenize = lambda x: nltk.wordpunct_tokenize(x.lower())
            elif language == 'zh':
                # borrow sacrebleu tokenizer since nltk does not support
                # chinese tokenization directly
                self._tokenizer = AutoTokenizer.from_pretrained('uer/t5-small-chinese-cluecorpussmall')
                self.tokenize = lambda x: self._tokenizer.decode(self._tokenizer.encode(x), skip_special_tokens=True).split()

            # Sentence level smoothing
            chencherry = nltk.translate.bleu_score.SmoothingFunction()
            self.smoothing_function = chencherry.method7

        elif pkg == 'sacrebleu':

            import sacrebleu
            from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
            from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

            if language == 'en':
                self._tokenizer = Tokenizer13a()
            elif language == 'zh':
                self._tokenizer = TokenizerZh()
            self.tokenize = lambda x: self._tokenizer(x).split(' ')

    def sentence_bleu(self, hyp:str=None, ref:List[str]=None, smooth=None):
        '''
        Argument:
            hyp (str): hypothesis as a string
            ref ([List[str]): list of references
        '''
        if self.pkg == 'sacrebleu':

            bleu_obj = sacrebleu.sentence_bleu(
                hypothesis=hyp,
                references=ref,
                tokenize='13a',
            )

            bleu_1, bleu_2, bleu_3, bleu_4 = bleu_obj.precisions
            bp = bleu_obj.bp  # brevity penalty

            return {
                'bleu_1': bp * (bleu_1),
                'bleu_2': bp * (bleu_1 * bleu_2) ** (1/2) ,
                'bleu_3': bp * (bleu_1 * bleu_2 * bleu_3) ** (1/3),
                'bleu_4': bp * (bleu_1 * bleu_2 * bleu_3 * bleu_4) ** (1/4),
                'bleu': bp * (bleu_1 * bleu_2 * bleu_3 * bleu_4) ** (1/4),
            }

        elif self.pkg == 'nltk':

            tokenized_hyps = self.tokenize(hyp)
            tokenized_refs = list(map(self.tokenize, ref))

            if smooth is None:
                smooth = self.smooth

            if smooth:
                sf = self.smoothing_function
            else:
                sf = None

            bleu_1 = 100 * nltk.translate.bleu_score.sentence_bleu(tokenized_refs, tokenized_hyps, weights=[1.0], smoothing_function=sf)
            bleu_2 = 100 * nltk.translate.bleu_score.sentence_bleu(tokenized_refs, tokenized_hyps, weights=[0.5, 0.5], smoothing_function=sf)
            bleu_3 = 100 * nltk.translate.bleu_score.sentence_bleu(tokenized_refs, tokenized_hyps, weights=[1/3, 1/3, 1/3], smoothing_function=sf)
            bleu_4 = 100 * nltk.translate.bleu_score.sentence_bleu(tokenized_refs, tokenized_hyps, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=sf)
            bleu   = bleu_4

        else:
            raise ValueError('Unknown package')

        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
            'bleu': bleu,
        }

    def corpus_bleu(self, hyps=None, refs=None, smooth=None):
        '''
        Argument:
            hyps List[str]: list of generating sentences
            refs List[List[str]]: list of list of references
        '''
        if self.pkg == 'sacrebleu':

            transposed_references = list(zip(*references))
            bleu_obj = sacrebleu.corpus_bleu(
                hypotheses=hyps,
                references=transposed_references,
                tokenize="13a",
                lowercase=True,
            )

            bleu_1, bleu_2, bleu_3, bleu_4 = bleu_obj.precisions
            bleu = bleu_obj.score

        elif self.pkg == 'nltk':

            if smooth is None:
                smooth = self.smooth

            if smooth:
                sf = self.smoothing_function
            else:
                sf = None

            tokenized_hyps = list(map(self.tokenize, hyps))
            tokenized_refs = [list(map(self.tokenize, ref)) for ref in refs]

            bleu_1 = 100 * nltk.translate.bleu_score.corpus_bleu(tokenized_refs, tokenized_hyps, weights=[1.0], smoothing_function=sf)
            bleu_2 = 100 * nltk.translate.bleu_score.corpus_bleu(tokenized_refs, tokenized_hyps, weights=[0.5, 0.5], smoothing_function=sf)
            bleu_3 = 100 * nltk.translate.bleu_score.corpus_bleu(tokenized_refs, tokenized_hyps, weights=[1/3, 1/3, 1/3], smoothing_function=sf)
            bleu_4 = 100 * nltk.translate.bleu_score.corpus_bleu(tokenized_refs, tokenized_hyps, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=sf)
            bleu   = bleu_4

        else:
            raise ValueError('Unknown package')

        return {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
            'bleu': bleu,
        }

    def dist(self, generations: List[str]):

        tokenized_gens = map(self.tokenize, generations)
        corpus = [token for tokenized_gen in tokenized_gens for token in tokenized_gen]

        bigram_finder = BigramCollocationFinder.from_words(corpus)
        try:
            bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N
        except ZeroDivisionError:
            print('Division by zero in dist-2 calcluation')
            bi_diversity = 0

        dist = FreqDist(corpus)

        try:
            uni_diversity = len(dist) / len(corpus)
        except ZeroDivisionError:
            print('Division by zero in dist-1 calcluation')
            uni_diversity = 0

        return uni_diversity * 100, bi_diversity * 100

def get_dfs(df_paths):
    dfs = []
    for df_path in df_paths:
        dfs.append(pd.read_csv(df_path, keep_default_na=False))
    return dfs

def calc_pairwise_bleu(outputs, metrics: Metrics):
    '''Given a list of outputs (str), calculate the pairwise BLEU
    '''
    pairwise_bleu = 0
    perms = list(itertools.permutations(range(len(outputs)), 2))
    for i, j in perms:
        pairwise_bleu += metrics.sentence_bleu(outputs[i], [outputs[j]])['bleu']
    return pairwise_bleu / len(perms)

def select_by_column(dfs, column='score', select_high=True):

    data = []

    num_dfs = len(dfs)
    num_generations = len(dfs[0])

    for i in range(num_generations):
        selected_idx = 0
        score = dfs[selected_idx].iloc[i][column]
        for j in range(1, num_dfs):
            if select_high:
                selected = dfs[j].iloc[i][column] > score
            else:
                selected = dfs[j].iloc[i][column] < score
            if selected:
                selected_idx = j
                score = dfs[j].iloc[i][column]
        data.append(list(dfs[selected_idx].iloc[i]))

    return pd.DataFrame(data, columns=dfs[0].columns)

def get_pairwise_bleu(dfs, metrics):

    num_dfs = len(dfs)
    num_generations = len(dfs[0])

    pairwise_bleus = []

    for i in range(num_generations):
        hyps = []
        for j in range(0, num_dfs):
            hyps.append(dfs[j].iloc[i]['output'])

        ## FIXME: sacrebleu cannot handle empty generations
        # if '' in hyps:
        #     print('[pairwise_bleu]: skipping test_{} due to empty generations'.format(i))
        # else:
        #     pairwise_bleus.append(calc_pairwise_bleu(hyps, metrics))

        pairwise_bleus.append(calc_pairwise_bleu(hyps, metrics))

    return statistics.mean(pairwise_bleus)

def get_confidence_acc(dfs):

    num_dfs = len(dfs)
    num_generations = len(dfs[0])

    corrects = []

    for i in range(num_generations):
        ppls = []
        bleus = []
        for j in range(0, num_dfs):
            ppls.append(dfs[j].iloc[i]['score'])
            bleus.append(dfs[j].iloc[i]['bleu'])
        min_ppl = min(ppls)
        max_bleu = max(bleus)

        if ppls.index(min_ppl) == bleus.index(max_bleu):
            corrects.append(1)
        else:
            corrects.append(0)

    return statistics.mean(corrects) * 100

def dist(generations: List[str], metrics: Metrics):
    '''Given a list of generations, return dist_1 and dist_2 metrics
    '''

    tokenized_gens = list(map(metrics.tokenize, generations))
    # bigrams and unigrams per generation
    nested_bigrams = map(lambda x: nltk.bigrams(x), tokenized_gens)
    nested_unigrams = tokenized_gens
    # pooled bigrams and unigrams
    bigrams = [bigram for bigrams in nested_bigrams for bigram in bigrams]
    unigrams = [unigram for unigrams in nested_unigrams for unigram in unigrams]

    bigram_fd = FreqDist(bigrams)
    unigram_fd = FreqDist(unigrams)

    if len(unigrams) == 0:
        dist_1 = 0
    else:
        dist_1 = len(unigram_fd) / len(unigrams) * 100

    if len(bigrams) == 0:
        dist_2 = 0
    else:
        dist_2 = len(bigram_fd) / len(bigrams) * 100

    return dist_1, dist_2

def get_inter_intra_dists(dfs, metrics):

    intradist_2s = []

    # interdists
    per_df_gens = [list(df['output']) for df in dfs]  #[i][j] -> ith df, jth sample
    pooled_gens = [gen for df_gens in per_df_gens for gen in df_gens]
    interdist_1, interdist_2 = dist(pooled_gens, metrics)

    # intradists
    per_sample_gens = list(map(list, zip(*per_df_gens)))  # "Transpose" of per_df_gens
    intradists = list(map(lambda x: dist(x, metrics), per_sample_gens))

    intradist_1s = list(map(lambda x: x[0], intradists))
    intradist_2s = list(map(lambda x: x[1], intradists))

    intradist_1 = statistics.mean(intradist_1s)
    intradist_2 = statistics.mean(intradist_2s)

    return interdist_1, interdist_2, intradist_1, intradist_2

def get_bleu_precision_recall(
    dfs: List[pd.DataFrame],
    references: List[str],
    metrics: Metrics):

    num_dfs = len(dfs)
    num_generations = len(dfs[0])

    precisions = [[] for _ in range(4)]
    recalls    = [[] for _ in range(4)]

    for i in range(num_generations):

        hypotheses = []

        for j in range(0, num_dfs):
            hypotheses.append(dfs[j].iloc[i]['output'])

        num_hyps = len(hypotheses)
        num_refs = len(references[i])
        sent_bleus = [np.zeros((num_hyps, num_refs)) for _ in range(4)]
        for hyp_i in range(num_hyps):
            for ref_j in range(num_refs):
                sent_bleu = metrics.sentence_bleu(
                    hyp=hypotheses[hyp_i],
                    ref=[references[i][ref_j]],
                )
                for bleu_i in range(4):
                    sent_bleus[bleu_i][hyp_i][ref_j] = sent_bleu['bleu_{}'.format(bleu_i + 1)]

        for bleu_i in range(4):
            precisions[bleu_i].append(sent_bleus[bleu_i].max(axis=1).mean())
            recalls[bleu_i].append(sent_bleus[bleu_i].max(axis=0).mean())

    results = dict()
    for bleu_i in range(4):
        results['bleu_{}_precision'.format(bleu_i + 1)] = statistics.mean(precisions[bleu_i])
        results['bleu_{}_recall'.format(bleu_i + 1)] = statistics.mean(recalls[bleu_i])

    return results

def get_bleu_precision_recall_parallel(
    dfs: List[pd.DataFrame],
    references: List[str],
    metrics: Metrics):

    # helper for multiprocessing
    def _precision_recall(x):
        hypotheses, references = x
        num_hyps = len(hypotheses)
        num_refs = len(references)
        sent_bleus = np.zeros((num_hyps, num_refs))
        for hyp_i in range(num_hyps):
            for ref_j in range(num_refs):
                sent_bleus[hyp_i][ref_j] = metrics.sentence_bleu(
                    hyp=hypotheses[hyp_i],
                    ref=[references[ref_j]],
                )['bleu']
        precision = sent_bleus.max(axis=1).mean()
        recall = sent_bleus.max(axis=0).mean()
        return precision, recall

    num_dfs = len(dfs)
    num_generations = len(dfs[0])

    per_df_gens = [list(df['output']) for df in dfs]  #[i][j] -> ith df, jth sample
    per_sample_gens = list(map(list, zip(*per_df_gens)))  # "Transpose" of per_df_gens

    # num_generations x 3,
    # first: list of generations (str), second: list of references (str), third: metrics
    inputs = list(zip(per_sample_gens, references))

    with mp.Pool(20) as p:
        mapped = list(tqdm(p.imap(_precision_recall, inputs), total=num_generations))

    precisions = list(map(lambda x: x[0], mapped))
    recalls = list(map(lambda x: x[1], mapped))

    return statistics.mean(precisions), statistics.mean(recalls)

def evaluate_single(df=None, inputs=None, references=None, metrics=None):
    '''
    Arguments:
        references[i][j] (List[List]) := the jth explanation for the ith sample
    Return:
        dict: metric -> value
            bleu-1,2,3,4
            bleu,
            ibleu,
            dist-1,2
    '''
    results = dict()
    generations = list(df['output'])

    bleu_obj = metrics.corpus_bleu(
        hyps=generations,
        refs=references,
    )
    # Record the BLEUs
    for metric, value in bleu_obj.items():
        results[metric] = value

    penalty_obj = metrics.corpus_bleu(
        hyps=generations,
        refs=[[i] for i in inputs],
    )
    # Record the iBLEUs
    for metric, value in penalty_obj.items():
        results['i' + metric] = 0.9 * bleu_obj[metric] - 0.1 * penalty_obj[metric]

    # ----- dist -----
    interdist_1, interdist_2, intradist_1, intradist_2 = get_inter_intra_dists([df], metrics)
    results['interdist_1'] = interdist_1
    results['interdist_2'] = interdist_2
    # # intra is not useful here for single generation...
    # results['intradist_1'] = intradist_1
    # results['intradist_2'] = intradist_2

    return results

def evaluate_single_multi(dfs, inputs, references, metrics):
    '''Evaluate multiple generations under the single generation setting
    '''
    min_ppl_df = select_by_column(dfs, column='score', select_high=False)
    results = evaluate_single(min_ppl_df, inputs, references, metrics)

    results['confidence_acc'] = get_confidence_acc(dfs)
    results['pairwise_bleu'] = get_pairwise_bleu(dfs, metrics)

    _, _, intradist_1, intradist_2 = get_inter_intra_dists(dfs, metrics)
    results['intradist_1'] = intradist_1
    results['intradist_2'] = intradist_2

    best_bleu_df = select_by_column(dfs, column='bleu', select_high=True)
    best_bleu_results = evaluate_single(best_bleu_df, inputs, references, metrics)

    for k, v in best_bleu_results.items():
        results['best_{}'.format(k)] = v

    return results

def evaluate_multi(dfs, inputs, references, metrics):

    results = {}

    # BLEU precision/recall
    bleu_results = get_bleu_precision_recall(dfs, references, metrics)
    for bleu_i in range(1, 5):
        precision = bleu_results['bleu_{}_precision'.format(bleu_i)]
        recall = bleu_results['bleu_{}_recall'.format(bleu_i)]
        results['bleu_{}_precision'.format(bleu_i)] = precision
        results['bleu_{}_recall'.format(bleu_i)] = recall
        results['bleu_{}_f1'.format(bleu_i)] = (2 * precision * recall) / (precision + recall)

    # inter/intra dists
    interdist_1, interdist_2, intradist_1, intradist_2 = get_inter_intra_dists(dfs, metrics)
    results['interdist_1'] = interdist_1
    results['interdist_2'] = interdist_2
    results['intradist_1'] = intradist_1
    results['intradist_2'] = intradist_2
    results['pairwise_bleu'] = get_pairwise_bleu(dfs, metrics)

    return results

def summarize(dfs, inputs, references, output_path):

    with open(output_path, mode='w') as f:

        num_dfs = len(dfs)
        num_generations = len(dfs[0])

        for i in range(num_generations):

            print('Input [{}]:'.format(i), inputs[i], file=f)

            for j in range(num_dfs):
                row = dfs[j].iloc[i]
                print('Output [{}], BLEU: {:6.2f}, PPL: {:6.2f}: {}'.format(j, row['bleu'], row['score'], row['output']), file=f)
            for k in range(len(references[i])):
                print('Reference [{}]: {}'.format(k, references[i][k]), file=f)

            print('', file=f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--eval-path', type=str, default='data/cleaned_ost/single-turn/test.csv')
    parser.add_argument('--generations-path', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--output-name', type=str, default='')
    # parser.add_argument('--generations-path', type=str, default='test_generate.csv')
    parser.add_argument('--mode', default='base', choices=['base', 'moe'])
    parser.add_argument('--multi-ref', action='store_true')
    parser.add_argument('--language', choices=['en', 'zh'], default='en')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--pkg', choices=['nltk', 'sacrebleu'], default='nltk')
    parser.add_argument('--single', action='store_true', help='If set, evaluate greedily')
    parser.add_argument('--target-beams', type=int, default=10)
    parser.add_argument('--do-nucleus', action='store_true')

    torch.manual_seed(0)

    args = parser.parse_args()

    inputs, references = default_dataset.get_inputs_and_references(
        args.eval_path,
        multi_ref=args.multi_ref,
    )

    metrics = Metrics(language=args.language, pkg=args.pkg, smooth=args.smooth)

    tzone = tz.gettz('America/Edmonton')
    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

    if args.ckpt_path:
        model = torch.load(args.ckpt_path).cuda()
        model.eval()
        if args.language == 'en':
            tokenizer = util.get_tokenizer('t5-small')
        else:
            tokenizer = util.get_tokenizer('uer/t5-small-chinese-cluecorpussmall')

    if args.output_dir and args.output_name:
        output_dir = args.output_dir
        output_name = args.output_name
    else:
        output_dir = PosixPath('gen_test')
        output_name = timestamp
        if not output_dir.exists():
            output_dir.mkdir()

    if args.generations_path:
        _dir, name = args.generations_path.rsplit('/', 1)
        df_paths = sorted(list(PosixPath(_dir).glob(name)))
        dfs = get_dfs(df_paths)

        # TODO: remove
        # cProfile.run('gold_precision, gold_recall = get_bleu_precision_recall(dfs, references, metrics)')
        # precision, recall = get_bleu_precision_recall_parallel(dfs, references, metrics)
        # interdist_1, interdist_2, intradist_1, intradist_2 = get_inter_intra_dists(dfs, metrics)

    else:

        generate_params = {}
        if args.do_nucleus:
            generate_params['do_sample'] = True
            generate_params['top_p'] = 0.95

        _dir, name = args.ckpt_path.rsplit('/', 1)

        if args.mode == 'base':

            dfs = generate.generate(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                references=references,
                output_dir=output_dir,
                output_name=output_name,
                metrics=metrics,
                num_beams=args.target_beams,
                num_return_sequences=args.target_beams,
                batch_size=32,
                **generate_params,
            )
            summarize(dfs, inputs, references, PosixPath(output_dir, output_name + '.sum'))

        elif args.mode == 'moe':
            dfs = []
            for mode_idx in range(model.num_modes):
                model.mode_idx = mode_idx
                output_path = PosixPath(args.ckpt_path + '.test.gen.{}'.format(mode_idx))
                df = generate.generate(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    references=references,
                    output_dir=output_dir,
                    output_name=output_name,
                    max_length=64,
                    metrics=metrics,
                    num_beams=args.target_beams // model.num_modes,
                    num_return_sequences=args.target_beams // model.num_modes,
                    batch_size=64,
                    **generate_params,
                )
                dfs += df

    if args.single:
        results = evaluate_single(dfs[0], inputs, references, metrics)
    else:
        results = evaluate_multi(dfs, inputs, references, metrics)

    with open(PosixPath(_dir, timestamp + '.eval'), mode='w') as f:

        print('> Command:', ' '.join(sys.argv))
        print('> Command:', ' '.join(sys.argv), file=f)

        for metric, value in results.items():
            print('{}: {}'.format(metric, value))
            print('{}: {}'.format(metric, value), file=f)

    print('done')
