class EncoderLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.norm_1 = Norm(opt.emb_dim)
        self.norm_2 = Norm(opt.emb_dim)

        self.dropout_1 = nn.Dropout(opt.dropout)
        self.dropout_2 = nn.Dropout(opt.dropout)

        self.attn = MultiHeadAttention(opt.heads, 
                                       opt.emb_dim)
        
        self.ff = FeedForward(opt)
        
        
    def forward(self, x, mask):
        '''
        This implementation follows the Tensor2Tensor implementation
        instead of the original paper "Attention is all you need"
        The Norm is applied to the input first, then self attention
        is applied to the sub-layer.
        '''

        x = self.norm_1(x)
        x1 = x + self.dropout_1(self.attn(x, x, x, mask))

        x1 = self.norm_2(x1)
        x2 = x1 + self.dropout_2(self.ff(x1))

        return x2
