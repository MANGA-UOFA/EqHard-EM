from transformers import AutoTokenizer, AutoModelWithLMHead


def get_tokenizer(model_str):

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    additional_tokens = {}

    if 'uer/t5-' in model_str:
        tokenizer.eos_token = tokenizer.sep_token

    if not tokenizer.eos_token:
        additional_tokens['eos_token'] = '<eos>'
    if not tokenizer.pad_token:
        additional_tokens['pad_token'] = '<pad>'
    if not tokenizer.sep_token:
        additional_tokens['sep_token'] = '<sep>'
    # Add special tokens for the model to recognize
    # tokenizer.add_tokens('__eou__')
    tokenizer.add_special_tokens(additional_tokens)

    return tokenizer
