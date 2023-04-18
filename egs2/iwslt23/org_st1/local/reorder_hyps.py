#!/usr/bin/env python3
# -*- encoding: utf8 -*-

"""
   TBD
"""

import argparse
import itertools
import os
import re
import sys
import yaml
from pathlib import Path
import xmltodict


parser = argparse.ArgumentParser()
parser.add_argument("--org", type=str)
parser.add_argument("--hyp", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    org = [x.strip() for x in open(args.org, "r").readlines()]
    org = [x[x.find('talkid')+len('talkid'):] for x in org]
    hyp = [x.strip() for x in open(args.hyp, "r").readlines()]

    hyp_dict = {}
    for h in hyp:
        try:
            uttid, txt = h.split(" ", 1)
        except:
            uttid = h
            txt = ""
        talkid = uttid.split("_")[1]
        if talkid in hyp_dict:
            hyp_dict[talkid].append(txt+"\n")
        else:
            hyp_dict[talkid] = [txt+"\n"]

    reordered = []
    for talkid in org:
        reordered += hyp_dict[talkid]

    with open(args.hyp+".reorder", "w") as f:
        f.writelines(reordered)