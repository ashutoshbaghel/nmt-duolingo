import math

import torch

from utils import nopeak_mask


class BeamSearch:

    def __init__(self, src, trg, model, opt):
        self.src = src
        self.trg = trg
        self.model = model
        self.opt = opt
        self.init_tok = trg.vocab.stoi['<sos>']

    def beam_search(self, encoded_seq, src_mask):
        assert encoded_seq.shape[0] == self.opt.k
        assert encoded_seq.shape[1] == self.opt.max_len

        eos_tok = self.trg.vocab.stoi['<eos>']
        outputs = torch.LongTensor([[self.init_tok]]).to(self.opt.device)
        ind = None

        for i in range(1, self.opt.max_len):
            trg_mask = nopeak_mask(i, self.opt).to(self.opt.device)

            if i == 1:
                out = self.model.decode(outputs,
                                        encoded_seq[0],
                                        src_mask,
                                        trg_mask)
            else:
                out = self.model.decode(outputs[:, :i],
                                        encoded_seq,
                                        src_mask,
                                        trg_mask)
            out = F.softmax(out, dim=-1)

            probs, ix = out[:, -1].data.topk(self.opt.k)

            if i == 1:
                log_scores = torch.Tensor([math.log(prob) \
                                           for prob in probs.data[0]]) \
                    .unsqueeze(0)
                outputs = torch.zeros(self.opt.k, opt.max_len).long().to(opt.device)

                outputs[:, 0] = self.init_tok
                outputs[:, 1] = ix[0]

                continue

            log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(self.opt.k, -1)
            log_scores.transpose(0, 1)

            k_probs, k_ix = log_probs.view(-1).topk(self.opt.k)

            outputs[:, :i] = outputs[k_ix // self.opt.k, :i]
            outputs[:, i] = ix[k_ix // self.opt.k, k_ix % self.opt.k]

            log_scores = k_probs.unsqueeze(0)

            sentence_lengths = torch.zeros(len(outputs),
                                           dtype=torch.long).to(self.opt.device)

            for vec in (outputs == eos_tok).nonzero():
                i = vec[0]
                if sentence_lengths[i] == 0:
                    sentence_lengths[i] = vec[1]

            num_finished_sentences = len([s for s in sentence_lengths \
                                          if s > 0])

            if num_finished_sentences == opt.k:
                alpha = self.opt.length_penalty
                div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            length = (outputs[0] == eos_tok).nonzero()
            return [self.trg.vocab.itos[tok] for tok in outputs[0]]
        else:
            length = (outputs[ind] == eos_tok).nonzero()[0]
            return [self.trg.vocab.itos[tok] for tok in outputs[ind][1:length]]

    def encode_batch(self, test_batch):
        src_mask = (test_batch != self.src.vocab.stoi['<pad>']).unsqueeze(-2)
        encoded_batch = self.model.encoder(test_batch, src_mask)
        return encoded_batch, src_mask.to(self.opt.device)
