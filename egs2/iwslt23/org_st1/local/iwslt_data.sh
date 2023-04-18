# # combine files
# src=$1
# dst=$2

# for f in text.lc.rm.en text.tc.de wav.scp; do
#     for c in mustc/dump/raw/train.en-de_sp/ mustc2/dump/raw/train.en-de_sp/ mustc3/dump/raw/train.en-de_sp/ stted/dump/raw/train_nodevtest_sp; do
#         cat "$src/$c/$f" >> "$dst/$f"
#     done
# done


# remove sp
src=$1
dst=$2

for f in text.lc.rm.en text.tc.de wav.scp; do
    cat "$src/$f" | grep -v -e "sp0.9" -e "sp1.1" >> "$dst/$f"
done