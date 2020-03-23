from torch import nn

from layers import DecoderLayer, PositionalEncoder, Embedder, EncoderLayer, Norm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, opt):
        super().__init__()
        self.N = opt.n_layers
        self.embed = Embedder(vocab_size, opt.emb_dim)
        self.pe = PositionalEncoder(opt)
        self.layers = get_clones(EncoderLayer(opt), self.N)
        self.norm = Norm(opt.emb_dim)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, opt):
        super().__init__()
        self.N = opt.n_layers
        self.embed = Embedder(vocab_size, opt.emb_dim)
        self.pe = PositionalEncoder(opt)
        self.layers = get_clones(DecoderLayer(opt), self.N)
        self.norm = Norm(opt.emb_dim)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)

        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, s_len, t_len, opt):
        super().__init__()
        assert opt.emb_dim % opt.heads == 0
        assert opt.dropout < 1

        self.opt = opt

        self.encoder = Encoder(s_len, self.opt)
        self.decoder = Decoder(t_len, self.opt)
        self.out = nn.Linear(opt.emb_dim, t_len)

    def forward(self, src_seq, trg_seq, src_mask, trg_mask):
        e_outputs = self.encoder(src_seq, src_mask)
        d_output = self.decoder(trg_seq, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

    def decode(self, decoder_input, encoder_output, src_mask, trg_mask):
        return self.out(model.decoder(decoder_input,
                                      encoder_output,
                                      src_mask,
                                      trg_mask))
