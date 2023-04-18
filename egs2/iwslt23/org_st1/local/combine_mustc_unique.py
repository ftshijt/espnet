#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--src", type=str, help="", required=True)
    parser.add_argument("--dst", type=str, help="", required=True)
    return parser

def main(args):
    args = get_parser().parse_args(args)

    src_ids = [x.split()[0] for x in open(args.src, "r").readlines()]

    mustc1_rec_ids = set()
    mustc2_rec_ids = set()
    mustc3_rec_ids = set()
    for line in src_ids:
        rec_id = line.split("_")[1]
        set_name = line.split("-")[0]
        if set_name == "MUSTC1":
            mustc1_rec_ids.add(rec_id)
        elif set_name == "MUSTC2":
            mustc2_rec_ids.add(rec_id)
        elif set_name == "MUSTC3":
            mustc3_rec_ids.add(rec_id)
        else:
            import pdb;pdb.set_trace()

    combined_ids = []
    for line in src_ids:
        rec_id = line.split("_")[1]
        set_name = line.split("-")[0]

        if set_name == "MUSTC1":
            if rec_id not in mustc3_rec_ids and rec_id not in mustc2_rec_ids:
                combined_ids.append(line+"\n")
        elif set_name == "MUSTC2":
            if rec_id not in mustc3_rec_ids:
                combined_ids.append(line+"\n")
        elif set_name == "MUSTC3":
            combined_ids.append(line+"\n")
        else:
            import pdb;pdb.set_trace()

    with open(args.dst, "w") as f:
        f.writelines(combined_ids)
            
    

if __name__ == "__main__":
    main(sys.argv[1:])