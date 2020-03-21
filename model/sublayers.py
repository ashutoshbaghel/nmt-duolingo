class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
    
        self.size = d_model
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, opt):
        super().__init__()

        linear_1 = nn.Linear(opt.emb_dim, opt.ff_hsize)
        dropout = nn.Dropout(opt.dropout)
        linear_2 = nn.Linear(opt.ff_hsize, opt.emb_dim)

        self.layers = nn.Sequential(linear_1, nn.ReLU(), dropout, linear_2)

    def forward(self, x):
        self.layers(x)
        return x
