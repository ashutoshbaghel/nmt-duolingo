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


# Train dataloader
train_iter = data_loader("train")

# Validation dataloader
val_iter = data_loader("dev")

# Test dataloader
test_iter = data_loader("test")
