import argparse
from pathlib import PosixPath
import util
import torch
import sys
import math
import numpy as np
import pandas as pd
import default_dataset
import evaluate_generations
from transformers.generation_utils import (
    # BeamSearchEncoderDecoderOutput,
    GreedySearchEncoderDecoderOutput,
    SampleEncoderDecoderOutput,
)

def generate(
    model=None,
    tokenizer=None,
    inputs=None,
    references=None,
    output_dir='.',
    output_name='noname',
    max_length=None,
    batch_size=64,
    metrics=None,
    num_beams=1,
    num_return_sequences=1,
    **gen_kwargs,
):
    assert model.training == False

    # shuffled due to sorting
    shuffled_outputs = [[] for _ in range(num_return_sequences)]
    shuffled_scores = [[] for _ in range(num_return_sequences)]
    shuffled_lengths = [[] for _ in range(num_return_sequences)]
    if references:
        shuffled_bleus = [[] for _ in range(num_return_sequences)]

    # first pass to re-order
    tokenized = tokenizer(inputs).input_ids
    lengths = [len(subl) for subl in tokenized]

    # sort in increasing order
    sorting_indices = np.argsort(lengths)
    reversing_indices = np.argsort(sorting_indices)

    sorted_inputs = [inputs[i] for i in sorting_indices]
    if references:
        sorted_references = [references[i] for i in sorting_indices]

    num_inputs = len(inputs)
    num_batches = math.ceil(num_inputs / batch_size)

    for batch_idx in range(num_batches):

        if batch_idx % 10 == 0:
            print('{}/{}'.format(batch_idx, num_batches))

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_inputs)

        minibatch_size = end_idx - start_idx
        minibatch = sorted_inputs[start_idx:end_idx]

        minibatch = list(map(lambda x: x + ' ' + tokenizer.eos_token, minibatch))

        minibatch_inputs = tokenizer(
            minibatch,
            max_length=None,
            truncation=False,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )

        generated = model.generate(
            input_ids=minibatch_inputs.input_ids.cuda(),
            attention_mask=minibatch_inputs.attention_mask.cuda(),
            bad_words_ids=[[tokenizer.unk_token_id]],
            output_scores=True,
            return_dict_in_generate=True,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            # no_repeat_ngram_size=1,
            # repetition_penalty=1.2,  # recommended in https://arxiv.org/pdf/1909.05858.pdf
            **gen_kwargs,
        )

        # shape for sequences := batch_size * num_RETURN_SEQUENCES, seq_len
        # shape for scores: seq[tuple] := batch_size * num_BEAMs, config.vocab_size)

        for i in range(minibatch_size * num_return_sequences):

            assert minibatch_size * num_return_sequences == generated.sequences.shape[0]

            i_sample = i // num_return_sequences
            i_beam   = i % num_return_sequences
            i_scores = i_sample * num_beams + i_beam

            data_idx = batch_idx * batch_size + i_sample

            sequence = generated.sequences[i][1:]
            length = int(sum(sequence != tokenizer.pad_token_id))
            sequence = sequence[:length]

            if 'sequences_scores' in generated:
                log_prob = generated.sequences_scores[i]
            # the Huggingface API is inconsistent about what "scores" is between different outputs
            elif isinstance(generated, GreedySearchEncoderDecoderOutput):

                if length == 0:
                    log_prob = torch.tensor(0)
                    print('Generated response w/ length 0')
                    print('Input: ', sorted_inputs[data_idx])
                    print('Generated: ', generated.sequences[i])
                else:
                    step_log_probs = []
                    for seq_idx in range(length):
                        token_idx = sequence[seq_idx]
                        step_log_prob = torch.log_softmax(generated.scores[seq_idx][i_scores], dim=0)[token_idx]
                        step_log_probs.append(step_log_prob)
                    log_prob = sum(step_log_probs) / length

            elif isinstance(generated, SampleEncoderDecoderOutput):
                # Placeholder for now
                log_prob = torch.tensor(0)
            else:
                raise NotImplementedError()

            # here, log_prob is already normalized
            score = float(torch.exp(-log_prob))

            generated_output = tokenizer.decode(sequence, skip_special_tokens=True)
            bleu = metrics.sentence_bleu(
                generated_output,
                sorted_references[data_idx],
            )['bleu']

            shuffled_scores[i_beam].append(score)
            shuffled_outputs[i_beam].append(generated_output)
            shuffled_lengths[i_beam].append(length)
            shuffled_bleus[i_beam].append(bleu)

    dfs = []

    for i_beam in range(num_return_sequences):

        ordered_scores = [shuffled_scores[i_beam][i] for i in reversing_indices]
        ordered_outputs = [shuffled_outputs[i_beam][i] for i in reversing_indices]
        ordered_lengths = [shuffled_lengths[i_beam][i] for i in reversing_indices]
        ordered_bleus = [shuffled_bleus[i_beam][i] for i in reversing_indices]

        df = pd.DataFrame(
            {
                'score': ordered_scores,
                'bleu': ordered_bleus,
                'length': ordered_lengths,
                'output': ordered_outputs,
            }
        )
        output_path = PosixPath(output_dir, output_name + '.beam_{}.gen'.format(i_beam))
        df.to_csv(output_path, index=False)
        dfs.append(df)
    return dfs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='')
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--output-name', type=str, default='noname')
    parser.add_argument('--language', choices=['en', 'zh'], default='en')
    args = parser.parse_args()

    inputs, references = default_dataset.get_inputs_and_references(args.input_path)

    model = torch.load(args.ckpt_path).cuda()
    tokenizer = util.get_tokenizer('t5-small')

    metrics = evaluate_generations.Metrics(language=args.language)

    generations = generate(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        references=references,
        output_dir=args.output_dir,
        output_name=args.output_name,
        max_length=64,
        metrics=metrics,
        batch_size=32,
        num_beams=10,
        num_return_sequences=10,
    )
