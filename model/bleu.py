from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu


class Score(object):
    def __init__(self, reference=None, hypothesis=None, style=None):
        self.hypothesis = hypothesis
        self.reference = reference
        self.style = style

    def calculate_score(self):
        if self.style is None or self.style is "corpus":
            return corpus_bleu(self.reference, self.hypothesis)
        elif self.style is "sentence":
            return sentence_bleu(self.reference, self.hypothesis)
