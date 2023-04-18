#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--src", type=str, help="", required=True)
    parser.add_argument("--uttids", type=str, help="", required=True)
    parser.add_argument("--dst", type=str, help="", required=True)
    return parser

def main(args):
    args = get_parser().parse_args(args)

    src_lines = [x for x in open(args.src, "r").readlines()]
    uttids = set([x.strip() for x in open(args.uttids, "r").readlines()])

    dst_lines = []
    for line in src_lines:
        uttid = line.split(" ")[0]
        if uttid in uttids:
            dst_lines.append(line)

    with open(args.dst, "w") as f:
        f.writelines(dst_lines)
            
    

if __name__ == "__main__":
    main(sys.argv[1:])