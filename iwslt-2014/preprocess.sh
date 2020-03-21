#!/usr/bin/env bash

echo 'Cloning the Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

src=de
tgt=en
prep=data-tokenized
output=data

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000
TRAIN=$output/train.en-de
BPE_CODE=$output/code


rm -f $TRAIN
for l in $src $tgt; do
    cat $prep/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $output/$f
    done
done

echo "generating vocab for both languages...."
for L in $src $tgt; do
    python $BPEROOT/get_vocab.py -i $output/train.$L -o $output/vocab.$TOKEN.$L
done
