from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def max_sentence_bleu(
    references,
    hypothesis,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    bleus = []
    for ref in references:
        bleus.append(sentence_bleu(
            [ref],
            hypothesis,
            weights=weights,
            smoothing_function=smoothing_function,
            auto_reweigh=auto_reweigh)
        )
    return max(bleus)

def max_corpus_bleu(
    list_of_references,
    hypotheses,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    max_list_of_references = []
    for references, hypothesis in zip(list_of_references, hypotheses):
        max_ref = []
        max_ref_bleu = -1  # guaranteed to be smaller than any bleu
        for ref in references:
            bleu = sentence_bleu(
                [ref],
                hypothesis,
                weights=weights,
                smoothing_function=smoothing_function,
                auto_reweigh=auto_reweigh
            )
            if bleu > max_ref_bleu:
                max_ref = [ref]
                max_ref_bleu = bleu
        max_list_of_references.append(max_ref)
    return corpus_bleu(
        max_list_of_references,
        hypotheses,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh
    )

def i_corpus_bleu(
    list_of_references,
    hypotheses,
    inputs,
    alpha=0.9,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    list_of_inputs = [[i] for i in inputs]
    bleu = corpus_bleu(
        list_of_references,
        hypotheses,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh,
    )
    penalty = corpus_bleu(
        list_of_inputs,
        hypotheses,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh,
    )
    return alpha * bleu - (1 - alpha) * penalty

def i_sentence_bleu(
    references,
    hypothesis,
    input_,
    alpha=0.9,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    bleu = sentence_bleu(
        references,
        hypothesis,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh
    )
    penalty = sentence_bleu(
        [input_],
        hypothesis,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh
    )
    return alpha * bleu - (1 - alpha) * penalty

if __name__ == '__main__':
    pass
