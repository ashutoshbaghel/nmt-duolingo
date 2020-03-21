from allennlp.training.learning_rate_schedulers.noam import NoamLR

def get_std_opt(model, optimizer):
    return NoamLR(optimizer,
                  opt.emb_dim,
                  opt.warmup,
                  opt.factor)
