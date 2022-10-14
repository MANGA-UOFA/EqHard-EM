import sys
import argparse
import torch
import statistics
import numpy as np
import itertools
import random
import util
# import sacrebleu

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from metrics.bleus import i_sentence_bleu, i_corpus_bleu
from transformers import AutoTokenizer, BertTokenizer, GPT2TokenizerFast
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from nltk import word_tokenize, wordpunct_tokenize
from esnli import build_esnli_tests


BLEU_WEIGHTS_MEAN = [
    [1.0],
    [0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.25, 0.25, 0.25, 0.25],
]

BLEU_WEIGHTS_SINGLE = [
    [1.0],
    [0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def list_drop_indices(input_list, drop_indices):
    return [item for idx, item in enumerate(input_list) if idx not in drop_indices]

def str_tokenize(sent):
    '''Given a sentence (str), return a list of tokenized characters
    '''
    return wordpunct_tokenize(sent)

def generate_fn(model, tokenizer, post, max_length):
    '''
    Arguments:
        model (MultiDecoderT5)
        post (str)
    Return:
        tuple (list<str>, list<float>): ([resp1, resp2, ...], [score1, score2, ...])
    '''

    if 'gpt2' in model.name_or_path or 'DialoGPT' in model.name_or_path:
        post += ' ' + tokenizer.sep_token

    input_ids = tokenizer.encode(post, return_tensors='pt').to(device)

    responses = []
    self_ppls = []

    max_output_length = max_length
    input_length = len(input_ids[0])

    if 'gpt2' in model.name_or_path or 'DialoGPT' in model.name_or_path:
        # gpt2 models' output includes input
        # therefore add the length of the input
        max_output_length += input_length

    generated = model.generate(
        input_ids=input_ids,
        # no_repeat_ngram_size=1,
        bad_words_ids=[[tokenizer.unk_token_id]],
        # repetition_penalty=1.2,  # recommended in https://arxiv.org/pdf/1909.05858.pdf
        output_scores=True,
        return_dict_in_generate=True,
        max_length=max_output_length,
        pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id,
    )

    if 'gpt2' in model.name_or_path or 'DialoGPT' in model.name_or_path:
        # for gpt2 models, generated sequences always start with the input
        sequence = generated.sequences[0][input_length:]
    else:
        # generated sequence always start with decoder_start_token_id, which we ignore here
        sequence = generated.sequences[0][1:]
    scores = generated.scores
    assert len(sequence) == len(scores)

    log_prob_sum = 0
    for t in range(len(sequence)):
        token_idx = sequence[t]
        log_prob_sum += torch.log_softmax(scores[t][0], dim=0)[token_idx]

    self_ppl = torch.exp(-log_prob_sum / len(sequence)).item()
    response = tokenizer.decode(sequence, skip_special_tokens=True)

    responses.append(response)
    self_ppls.append(self_ppl)

    return responses, self_ppls

def gen_from_text(model, tokenizer, post, max_length):
    global gen_i
    gen_i += 1
    return [gen_responses[gen_i - 1]], [0]

def calc_pairwise_bleu(hyps):
    '''Given a list of hypothesis, calculate the pairwise BLEU
    '''
    pairwise_bleu = 0
    perms = list(itertools.permutations(range(len(hyps)), 2))
    for i, j in perms:
        pairwise_bleu += sentence_bleu([hyps[i]], hyps[j])
    return pairwise_bleu / len(perms)

def calculate_ngram_diversity(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
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

    return uni_diversity, bi_diversity

def eval_model(
        tests,
        model,
        tokenizer,
        generate_func=generate_fn,
        thresholds=[1.0],
        stream=None,
        num_dist_samples=None,
        max_length=64,
    ):
    '''
    Arguments:
        tests (list): a list of namedtuples, each containing score (float),
            context (str), and a list of responses (str)
        generate_func (lambda): function that takes a post (str) as input
            and generates a list of responses (list<str>) and their confidences (float)
    Return:
        dict: metric (str) -> value (float)
    '''

    # model could be a dummy variable
    if model:
        assert not model.training

    def _log(*args):
        if stream:
            print(*args, file=stream)
        else:
            print(*args)

    def calc_results(
        sent_bleu_1s,
        sent_bleu_2s,
        sent_bleu_3s,
        sent_bleu_4s,
        sent_ibleu_1s,
        sent_ibleu_2s,
        sent_ibleu_3s,
        sent_ibleu_4s,
        corp_inps,
        corp_refs,
        corp_model_hyps,
        corp_best_hyps,
        prefix_str='',
    ):

        # print(i_corpus_bleu(corp_refs, corp_best_hyps, corp_inps))
        _log('sent_bleus (1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
            statistics.mean(sent_bleu_1s),
            statistics.mean(sent_bleu_2s),
            statistics.mean(sent_bleu_3s),
            statistics.mean(sent_bleu_4s),
        ))
        _log('sent_ibleus (1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
            statistics.mean(sent_ibleu_1s),
            statistics.mean(sent_ibleu_2s),
            statistics.mean(sent_ibleu_3s),
            statistics.mean(sent_ibleu_4s),
        ))
        _log()

        corp_model_bleu1 = corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[0])
        corp_model_bleu2 = corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[1])
        corp_model_bleu3 = corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[2])
        corp_model_bleu4 = corpus_bleu(corp_refs, corp_model_hyps, weights=BLEU_WEIGHTS_MEAN[3])
        _log('corp_model_bleus(1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
            corp_model_bleu1,
            corp_model_bleu2,
            corp_model_bleu3,
            corp_model_bleu4,
        ))

        corp_model_ibleu1 = i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[0])
        corp_model_ibleu2 = i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[1])
        corp_model_ibleu3 = i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[2])
        corp_model_ibleu4 = i_corpus_bleu(corp_refs, corp_model_hyps, corp_inps, weights=BLEU_WEIGHTS_MEAN[3])
        _log('corp_model_ibleus(1-4): {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
            corp_model_ibleu1,
            corp_model_ibleu2,
            corp_model_ibleu3,
            corp_model_ibleu4,
        ))
        _log()

        if num_dist_samples:
            dist_hyps = random.sample(corp_model_hyps, k=num_dist_samples)
        else:
            dist_hyps = corp_model_hyps
        tokens = [token for sentence in dist_hyps for token in sentence]
        dist_1, dist_2 = calculate_ngram_diversity(tokens)
        _log('dist_1: {:.5f}, dist_2: {:.5f}'.format(dist_1, dist_2))

        _log()

        # eval_ as prefix for huggingface logger to understand that this is eval...
        return {
            '{}corp_model_bleu1'.format(prefix_str): corp_model_bleu1,
            '{}corp_model_bleu2'.format(prefix_str): corp_model_bleu2,
            '{}corp_model_bleu3'.format(prefix_str): corp_model_bleu3,
            '{}corp_model_bleu4'.format(prefix_str): corp_model_bleu4,
            '{}corp_model_ibleu1'.format(prefix_str): corp_model_ibleu1,
            '{}corp_model_ibleu2'.format(prefix_str): corp_model_ibleu2,
            '{}corp_model_ibleu3'.format(prefix_str): corp_model_ibleu3,
            '{}corp_model_ibleu4'.format(prefix_str): corp_model_ibleu4,
            '{}dist_1'.format(prefix_str): dist_1,
            '{}dist_2'.format(prefix_str): dist_2,
        }

    final_results = {}

    chosen_count = np.zeros(1)

    sent_bleu_1s  = []
    sent_bleu_2s  = []
    sent_bleu_3s  = []
    sent_bleu_4s  = []

    sent_ibleu_1s = []
    sent_ibleu_2s = []
    sent_ibleu_3s = []
    sent_ibleu_4s = []

    corp_refs = []  # List[List[List(str)]]
    corp_inps = []  # List[List(str)], list of inputs for iBLEU calcluation

    corp_model_hyps = []  # List[List(str)], list of hypothesis (list of chars)
    corp_best_hyps = []  # List[List(str)], list of hypothesis (list of chars)

    # sacrebleu
    sacre_model_sys = []  # List[str]
    sacre_best_sys = []

    sacre_exp1s = []
    sacre_exp2s = []

    num_posts = len(tests)

    correct_predictions = 0

    for i in range(num_posts):

        gold_label, context, reference_responses = tests[i]

        generated_responses, self_ppls = generate_func(model, tokenizer, context, max_length=max_length)

        inp = str_tokenize(context)
        corp_inps.append(inp)

        ref = list(map(lambda x: str_tokenize(x), reference_responses))
        corp_refs.append(ref)
        sacre_exp1s.append(reference_responses[0])
        sacre_exp2s.append(reference_responses[1])

        # for finding the response that the model is most confident with
        model_response = ''
        lowest_ppl = float('inf')
        chosen_idx = -1

        # for finding the response that works the best
        best_response = ''
        highest_bleu = -1

        _log('{}/{} - Post: {}'.format(i, num_posts - 1, ' '.join(inp)))

        # predict neutral by default
        final_prediction = 'neutral'

        # ----- deal with generated response for each decoder -----
        for j in range(len(generated_responses)):

            generated_response = generated_responses[j]
            if len(generated_response.split()) == 1:
                prediction = generated_response
                generated_response = ''
            else:
                prediction, generated_response = generated_response.split(' ', 1)

            self_ppl = self_ppls[j]

            hyp = str_tokenize(generated_response)

            sent_bleu_1 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[0])
            sent_bleu_1s.append(sent_bleu_1)
            sent_bleu_2 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[1])
            sent_bleu_2s.append(sent_bleu_2)
            sent_bleu_3 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[2])
            sent_bleu_3s.append(sent_bleu_3)
            sent_bleu_4 = sentence_bleu(ref, hyp, weights=BLEU_WEIGHTS_MEAN[3])
            sent_bleu_4s.append(sent_bleu_4)

            sent_ibleu_1 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[0])
            sent_ibleu_1s.append(sent_ibleu_1)
            sent_ibleu_2 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[1])
            sent_ibleu_2s.append(sent_ibleu_2)
            sent_ibleu_3 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[2])
            sent_ibleu_3s.append(sent_ibleu_3)
            sent_ibleu_4 = i_sentence_bleu(ref, hyp, inp, weights=BLEU_WEIGHTS_MEAN[3])
            sent_ibleu_4s.append(sent_ibleu_4)

            bleu = sent_bleu_4

            if bleu > highest_bleu:
                highest_bleu = bleu
                best_response = generated_response
            if self_ppl < lowest_ppl:
                lowest_ppl = self_ppl
                model_response = generated_response
                chosen_idx = j
                if prediction == 'contradiction' or prediction == 'entailment':
                    final_prediction = prediction

            _log('Response #{}, bleu={:.5f}, self_ppl={:9.2f}: {}'.format(j, bleu, self_ppl, generated_response))
            _log('Ref Response: {}'.format(reference_responses[0]))
            _log('gold_label: {}, predicted: {}'.format(gold_label, final_prediction))

        chosen_count[chosen_idx] += 1

        if prediction == gold_label:
            correct_predictions += 1

        _log()

        corp_model_hyps.append(str_tokenize(model_response))
        sacre_model_sys.append(model_response)
        corp_best_hyps.append(str_tokenize(best_response))
        sacre_best_sys.append(best_response)

    _log('---------- Results ----------')
    _log()

    results = calc_results(
        sent_bleu_1s,
        sent_bleu_2s,
        sent_bleu_3s,
        sent_bleu_4s,
        sent_ibleu_1s,
        sent_ibleu_2s,
        sent_ibleu_3s,
        sent_ibleu_4s,
        corp_inps,
        corp_refs,
        corp_model_hyps,
        corp_best_hyps,
        prefix_str='eval/'
    )

    prediction_acc = correct_predictions / num_posts * 100
    results['prediction_acc'] = prediction_acc
    _log('Prediction acc: {:.2f}'.format(prediction_acc))

    sacre_model_bleu = sacrebleu.corpus_bleu(sacre_model_sys, [sacre_exp1s, sacre_exp2s],
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False
    )
    sacre_best_bleu = sacrebleu.corpus_bleu(sacre_best_sys, [sacre_exp1s, sacre_exp2s],
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False
    )
    _log('sacre_model_bleu: {:.2f}'.format(sacre_model_bleu.score))
    _log('sacre_best_bleu:  {:.2f}'.format(sacre_best_bleu.score))

    _log('')
    results['sacre_model_bleu'] = sacre_model_bleu.score
    results['sacre_best_bleu']  = sacre_best_bleu.score

    return results

if __name__ == '__main__':

    # For sampling in diversity calculations
    random.seed(0)

    parser = argparse.ArgumentParser('Script for evaluating models')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--output-file', type=str, default='')
    parser.add_argument('--eval-path', type=str, default='data/esnli/esnli_test.csv')
    parser.add_argument('--max-num-dialogues', type=int, default=None)
    parser.add_argument('--gen-from-text', action='store_true')
    parser.add_argument('--generated-path', type=str, default='')
    parser.add_argument('--tokenizer-str', type=str, default='t5-small')

    args = parser.parse_args()

    if args.gen_from_text:
        assert args.generated_path
        with open(args.generated_path, mode='r') as f:
             gen_responses = f.readlines()
             gen_i = 0
        model = None
        tokenizer = None
    else:
        model = torch.load(args.ckpt, map_location=device)
        model.eval()
        tokenizer = util.get_tokenizer('t5-small')

    tests = build_esnli_tests(path=args.eval_path)

    if args.output_file:
        stream = open(args.output_file, mode='w')
    elif args.gen_from_text:
        stream = open(args.generated_path + '.test', mode='w')
    else:
        stream = open(args.ckpt + '.test', mode='w')

    if args.gen_from_text:
        print(eval_model(tests, model, tokenizer, stream=stream, thresholds=[0.25, 0.50, 0.75, 1.00], generate_func=gen_from_text))
    else:
        print(eval_model(tests, model, tokenizer, stream=stream, thresholds=[1.00]))
