import torch
from torch import nn
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = torch.triu(torch.ones((1, size, size)), diagonal=1)
    np_mask = Variable(np_mask == 0)
    return np_mask


def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size, opt)
        np_mask = np_mask.to(opt.device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask


def init_model_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def s_key(lst):
    return lst[1]


def append_to_list(output):
    print(output)
    for out in output:
        line = ' '.join(out) \
            .replace('@@ ', '') \
            .replace('<sos>', '') \
            .replace('<eos>', '') \
            .replace('<unk>', '')
        hypothesis.append(line)
