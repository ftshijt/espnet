#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
from collections import defaultdict, OrderedDict


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--src", type=str, help="", required=True)

    return parser

def main(args):
    args = get_parser().parse_args(args)

    lines = [json.loads(x.strip()) for x in open(args.src, "r").readlines()]

    increment_counts = defaultdict(int)
    total = 0
    for line in lines:
        local_counts = defaultdict(int)
        for word_delay in line['delays']:
            local_counts[word_delay] += 1
            
        for increment in local_counts.keys():
            increment_counts[local_counts[increment]] += 1
            total += 1
    
    for k in increment_counts.keys():
        increment_counts[k] = increment_counts[k] / total

    increment_counts = OrderedDict(sorted(increment_counts.items()))
    import pdb;pdb.set_trace()

    

if __name__ == "__main__":
    main(sys.argv[1:])