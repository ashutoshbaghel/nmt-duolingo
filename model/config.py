class Configuration(object):
  def __init__(self, source, target):
    self.src_data = source
    self.trg_data = target
    self.src_lang = 'de'
    self.trg_lang = 'en'
    self.epochs = 50
    self.n_layers = 6
    self.heads = 8
    self.dropout = 0.1
    self.printevery = 10
    self.lr = 5e-4
    self.emb_dim = 512
    self.ff_hsize = 1024
    self.max_strlen = 80
    self.checkpoint = 0
    self.device = 0
    self.clip_norm = 0.0
    self.src_pad = src.vocab.stoi['<pad>']
    self.trg_pad = trg.vocab.stoi['<pad>']
    self.k = 10
    self.max_len = 80
    self.device = torch.device("cuda:0" if use_cuda else "cpu")
