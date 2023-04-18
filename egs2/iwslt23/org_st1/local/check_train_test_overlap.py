#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train", type=str, help="", required=True)
    parser.add_argument("--test", type=str, help="", required=True)
    return parser

def main(args):
    args = get_parser().parse_args(args)

    train_ids = [x.split()[0] for x in open(args.train, "r").readlines()]
    test_ids = [x.split()[0] for x in open(args.test, "r").readlines()]

    train_rec_ids = set()
    for line in train_ids:
        rec_id = line.split("_")[1]

        set_name = line.split("-")[0]
        if set_name == "STTED":
            rec_id = "0" + rec_id

        assert len(rec_id) == 5
        train_rec_ids.add(rec_id)

    test_rec_ids = set()
    for line in test_ids:
        rec_id = line.split("_")[1]
        if len(rec_id) < 5:
            rec_id = "0" + rec_id

        assert len(rec_id) == 5
        test_rec_ids.add(rec_id)

    overlap = test_rec_ids.intersection(train_rec_ids)
    print("number of overlaps: " + str(len(overlap)))
    print("total recordings: " + str(len(test_rec_ids)))
    print(overlap)
            
    

if __name__ == "__main__":
    main(sys.argv[1:])