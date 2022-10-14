from collections import OrderedDict
import torch
import copy
import argparse

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

class MultiDecoderT5(T5ForConditionalGeneration):

    def __init__(self, config, num_modes=3):
        super().__init__(config)
        self.num_modes = num_modes
        self._mode_idx = 0
        self.decoders = torch.nn.ModuleList([copy.deepcopy(self.decoder) for _ in range(self.num_modes)])
        del self.decoder

    @property
    def decoder(self):
        return self.decoders[self.mode_idx]

    @property
    def mode_idx(self):
        return self._mode_idx

    @mode_idx.setter
    def mode_idx(self, value):
        self._mode_idx = value

    def from_single_pretrained(self, state_dict):

        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if 'decoder' in k:
                for i in range(self.num_modes):
                    new_sd[k.replace('decoder', 'decoders.{}'.format(i))] = v
            else:
                new_sd[k] = v

        self.load_state_dict(new_sd, strict=True)

    def encode(self, input_ids=None, attention_mask=None):
        '''Takes in input ids and return the encoder outputs for generation and loss
        '''
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return encoder_outputs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--init-ckpt', type=str, required=True)
    parser.add_argument('--num-modes', type=int, default=3)

    args = parser.parse_args()


    # loading the model
    init_ckpt = torch.load(args.init_ckpt)
    config = init_ckpt.config
    model = MultiDecoderT5(config, num_modes=args.num_modes)
    model.from_single_pretrained(init_ckpt.state_dict())
    model.train()
    del init_ckpt

    model = MultiDecoderT5(config)

    print('done!')
