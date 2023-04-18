src=data/tst2020/IWSLT.TED.tst2020.en-de.en.xml
tgt=data/tst2020/IWSLT.TED.tst2020.en-de.de.xml
ord=data/tst2020/FILE_ORDER
hyp=$1/text
dst=$1/mwer_results.txt

python3 local/reorder_hyps.py --hyp $hyp --org $ord
segmentBasedOnMWER.sh $src $tgt $hyp.reorder st German $hyp.reorder.xml normalize 1
sed -e "/<[^>]*>/d" $hyp.reorder.xml > $hyp.reorder.reseg
detokenizer.perl -l de -q < $hyp.reorder.reseg > $hyp.reorder.reseg.detok

sacrebleu data/tst2020/text.de -i $hyp.reorder.reseg.detok -m bleu > $dst
cat $dst
