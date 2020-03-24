from collections import Iterator

from torchtext.data import BucketIterator
from torchtext.data import Field
from torchtext.datasets import TranslationDataset

from config import Configuration
from optim import get_std_opt
from transformer import *
from translate import *
from utils import *


def data_loader(mode):
    data = TranslationDataset(
        path=dir_path + f"/{mode}", exts=('.de', '.en'),
        fields=(src, trg))

    iterator = None
    if mode == "train" or mode == "dev":
        if mode == "train":
            src.build_vocab(data)
            trg.build_vocab(data)

        iterator = BucketIterator(dataset=data, batch_size=128,
                                  sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

    else:
        iterator = Iterator(dataset=data, batch_size=128, train=False,
                            shuffle=False, sort=False)

    return iterator


if __name__ == '__main__':
    mode = "train"
    saved_checkpoint_path = "/content/drive/My Drive/Spring-20/11-747/11-747 Project/checkpoints/afternorm-01-02_25_12/transformer_5_model.pth"
    check_dir = '/content/drive/My Drive/Spring-20/11-747/11-747 Project/checkpoints'

    ## Load Vocab
    dir_path = '/content/drive/My Drive/Spring-20/11-747/11-747 Project/de-en/data/fairseq_data'
    src = Field(fix_length=100)
    trg = Field(init_token="<sos>", eos_token="<eos>", fix_length=100)

    # Model Configuration
    opt = Configuration(src, trg)

    ## Dataloaders
    # Train dataloader
    opt.train_iter = data_loader("train")

    # Validation dataloader
    opt.val_iter = data_loader("dev")

    # Test dataloader
    opt.test_iter = data_loader("test")

    transformer = Transformer(len(src.vocab), len(trg.vocab), opt)
    model = init_model_params(transformer)
    model = model.to(opt.device)

    opt.optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     betas=(0.9, 0.98),
                                     weight_decay=0.0001,
                                     eps=1e-9)

    opt.scheduler = get_std_opt(model, opt)

    translator = Translator(model, opt)

    if mode == "train":
        check_path = translator.make_checkpoint_dir(check_dir)
        translator.train(check_path)
    elif mode == "test":
        translator.load_checkpoint(saved_checkpoint_path, model)
        translator.test()
