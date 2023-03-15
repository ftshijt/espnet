import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str)

args = parser.parse_args()

for task in ["IDF", "IDM", "CDF", "CDM"]:
    wavscp = os.path.join(args.submission, task, "{}-wav.scp".format())
    wav_path = os.path.join(args.submission, task)
    base = 30000
    for i in range(24):
        wavscp.write("SF_{} {}/SF_{}.wav".format(base + i, wav_path, base + i))
        wavscp.write("SM_{} {}/SM_{}.wav".format(base + i, wav_path, base + i))
    wavscp.close()
