import os
import time
from datetime import datetime

import pytz
import torch
import torch.nn.functional as F

from search import BeamSearch
from utils import create_masks, s_key, append_to_list


def translate_sentence(test_batch, b_search, opt):
    final_output = []
    encoded_batch, src_masks = b_search.encode_batch(test_batch)

    for i, (encoded_seq, src_mask) in enumerate(zip(encoded_batch, src_masks)):
        encoded_seqs = torch.zeros(opt.k,
                                   encoded_seq.size(-2),
                                   encoded_seq.size(-1))
        encoded_seqs[:, :] = encoded_seq[0]
        encoded_seqs = encoded_seqs.to(opt.device)

        sentence = b_search.beam_search(encoded_seqs, src_mask)
        final_output.append(sentence)

        del encoded_seqs

    del encoded_batch
    del src_masks

    return final_output


class Translator:
    def __init__(self, model, opt):
        self.opt = opt
        self.model = model

    def train(self, check_path):
        print("training model...")
        opt = self.opt
        model = self.model

        model.train()
        start = time.time()

        best_acc = []
        step = 0

        for epoch in range(opt.epochs):
            total_loss = 0

            for i, batch in enumerate(opt.train_iter):
                src_seq = batch.src.transpose(0, 1)
                trg_seq = batch.trg.transpose(0, 1)

                src_seq, trg_seq = src_seq.to(self.opt.device), trg_seq.to(self.opt.device)
                trg_input = trg_seq[:, :-1].to(self.opt.device)

                src_mask, trg_mask = create_masks(src_seq, trg_input, opt)

                src_mask = src_mask.to(self.opt.device)
                trg_mask = trg_mask.to(self.opt.device)

                preds = model(src_seq, trg_input, src_mask, trg_mask)

                ys = trg_seq[:, 1:].contiguous().view(-1)
                opt.optimizer.zero_grad()

                loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                       ys,
                                       ignore_index=opt.trg_pad)
                loss.backward()

                opt.optimizer.step()
                opt.scheduler.step_batch(step)

                total_loss += loss.item()
                step += 1

                if (i + 1) % opt.printevery == 0:
                    p = int(100 * (i + 1) / len(opt.train_iter))
                    print(f"total loss: {total_loss}")
                    avg_loss = total_loss / opt.printevery
                    elapsed_time = int((time.time() - start) // 60)
                    print(f"{elapsed_time}m | epoch {epoch} | ",
                          f"{p}% | loss = {avg_loss}")
                    total_loss = 0

                del src_seq
                del trg_seq
                del src_mask
                del trg_mask

            val_loss = self.eval(model)
            self.best_checkpoints(best_acc, val_loss, epoch, check_path)
            self.save_checkpoint(epoch, model, opt, check_path)

            print(f"{(time.time() - start) // 60}m:  | epoch {epoch + 1}  100% \
                    loss = {avg_loss:.3f}\nepoch {epoch + 1} | complete, \
                    loss = {avg_loss:.3f}, val_loss = {val_loss:0.3f}")

    def eval(self, model):
        model = self.model
        model.eval()
        total_loss = 0

        for i, batch in enumerate(self.opt.val_iter):
            with torch.no_grad():
                src = batch.src.transpose(0, 1)
                trg = batch.trg.transpose(0, 1)
                src, trg = src.to(self.opt.device), trg.to(self.opt.device)
                trg_input = trg[:, :-1]

                src_mask, trg_mask = create_masks(src, trg_input, self.opt)

                src_mask = src_mask.to(self.opt.device)
                trg_mask = trg_mask.to(self.opt.device)

                preds = model(src, trg_input, src_mask, trg_mask)
                ys = trg[:, 1:].contiguous().view(-1)

                loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                       ys, ignore_index=self.opt.trg_pad)

                total_loss += loss.item()

                del src
                del trg
                del src_mask
                del trg_mask

        val_loss = total_loss / len(self.opt.val_iter)
        model.train()
        return val_loss

    def test(self):
        print("testing model...")
        self.model.eval()
        b_search = BeamSearch(self.opt.src, self.opt.trg, self.model, self.opt)

        for i, batch in enumerate(self.opt.test_iter):
            test_batch = batch.src.transpose(0, 1).to(self.opt.device)
            output = translate_sentence(test_batch, b_search, self.opt)
            append_to_list(output)

    def best_checkpoints(self, best_acc, val_loss, epoch, check_path):
        best_acc.append((epoch, val_loss))

        with open(os.path.join(check_path, "stats.txt"), "a+") as fl:
            size = 5 if len(best_acc) > 4 else len(best_acc)
            best = sorted(best_acc, key=s_key)[:size]
            indices = [str(a[0]) for a in best]
            pt = " ".join(indices)
            fl.write(f"epoch_num: {epoch}, val_loss: {val_loss}, \
                    top 5 checkpoints: {pt}\n")
            fl.write("====\n")
            fl.close()

        return

    def make_checkpoint_dir(self, path):
        d = datetime.now()
        EST = pytz.timezone('US/Eastern')
        d = d.astimezone(EST)
        fd = str(d.strftime("afternorm-%d-%H_%M_%S"))

        check_path = os.path.join(path, fd)

        try:
            os.mkdir(check_path)
        except OSError:
            print("Creation of the directory %s failed" % check_path)
        else:
            print("Successfully created the directory %s " % check_path)
        return check_path

    def save_checkpoint(self, epoch, model, opt, check_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.optimizer.state_dict(),
        },
            os.path.join(check_path,
                         'transformer_' + str(epoch) + '_model.pth'))

    def load_checkpoint(self, checkpoint_path, model):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        model = model.to(self.opt.device)
        self.model = model
