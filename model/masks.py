def nopeak_mask(size, opt):
    np_mask = torch.triu(torch.ones((size, size + 1)), diagonal=1)
    np_mask =  Variable(np_mask == 0)
    return np_mask


def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad)
        size = trg.size(1)
        np_mask = nopeak_mask(size, opt)
    else:
        trg_mask = None
        
    return src_mask, trg_mask, np_mask
