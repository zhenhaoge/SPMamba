# Combine two json files
#
# Zhenhao Ge, 2025-03-06

import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-infile1', type=str, help='json input file 1')
    parser.add_argument('--json-infile2', type=str, help='json input file 2')
    parser.add_argument('--json-outfile', type=str, help='json output file')
    args = parser.parse_args()
    return args

def main(infile1, infile2, outfile):

    # read from json file 1
    with open(infile1, "r") as f:
        infos1 = json.load(f)
    
    # read from json file 2
    with open(infile2, "r") as f:
        infos2 = json.load(f)

    infos = infos1 + infos2
    with open(outfile, "w") as f:
        json.dump(infos, f, indent=4)
    print(f'wrote json file: {outfile}')    


if __name__ == "__main__":

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_dir = os.getcwd()
    # args.json_infile1 = os.path.join(work_dir, 'data', 'WSJ0-2Mix', '16k', 'test', 'mix.json')
    # args.json_infile2 = os.path.join(work_dir, 'data', 'Echo2Mix', '16k', 'test', 'mix.json')
    # args.json_outfile = os.path.join(work_dir, 'data', 'WSJ0-Echo-2Mix', '16k', 'test', 'mix.json')

    # check file existence
    assert os.path.isfile(args.json_infile1), f'json file1: {args.json_infile1} does not exist!'
    assert os.path.isfile(args.json_infile2), f'json file2: {args.json_infile1} does not exist!'

    # localize files
    infile1 = args.json_infile1
    infile2 = args.json_infile2
    outfile = args.json_outfile

    # get output dir
    out_dir = os.path.dirname(outfile)

    # create output dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    main(infile1, infile2, outfile)    