###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2022-09-26 11:14:20
###

#%%
import os
import random
from typing import Union
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
import shutil
import scipy.io.wavfile as wav
from datetime import datetime
# import torchaudio
warnings.filterwarnings("ignore")
import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import (
    tensors_to_device,
    RichProgressBarTheme,
    MyMetricsTextColumn,
    BatchesProcessedColumn
)
from look2hear.system import make_optimizer
from look2hear.utils.parser_utils import str2bool
from look2hear.utils import print_only

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

def write_outputs(test_set, idx, est_sources_np, ex_save_dir, dataset=None):
    """write 5 wav files per entry:
        - original clean speaker file 1 and 2 before mixing (2 files)
        - mixed file (1 file)
        - separated speaker file 1 and 2 from the mixed file (2 files)
    """

    # get 3 input files
    mix_file = test_set.mix[idx][0]
    spk1_reverb_file = test_set.sources[0][idx][0]
    spk2_reverb_file = test_set.sources[1][idx][0]

    # set dataset (optional)
    if not dataset:
        parts = mix_file.split(os.sep)
        if 'WSJ0' in parts:
            dataset = 'WSJ0-2Mix'
        elif 'Echo2Mix' in parts:
            dataset = 'Echo2Mix'
        else:
            raise Exception('dataset is not set!')

    # get unique prefix
    ndigits = len(str(len(test_set)))
    if dataset == 'Echo2Mix':
        record, room, spks, _ = mix_file.split(os.sep)[-4:]
        prefix = '-'.join([f'{idx:0{ndigits}d}', record, room, spks])
    elif dataset == 'WSJ0-2Mix':
        filename = os.path.splitext(os.path.basename(mix_file))[0]
        prefix = '-'.join([f'{idx:0{ndigits}d}', filename])

    # define 3 output files
    mix_file2 = os.path.join(ex_save_dir, f'{prefix}-mix.wav')
    spk1_reverb_file2 = os.path.join(ex_save_dir, f'{prefix}-s1_reverb.wav')
    spk2_reverb_file2 = os.path.join(ex_save_dir, f'{prefix}-s2_reverb.wav')

    # write 3 output files (just for reference)
    if test_set.normalize_audio:
        # apply normalization
        mixture, sources, filename = test_set.preprocess_audio_only(idx)
        wav.write(mix_file2, test_set.sample_rate, mixture.cpu().numpy())
        wav.write(spk1_reverb_file2, test_set.sample_rate, sources[0].cpu().numpy())
        wav.write(spk2_reverb_file2, test_set.sample_rate, sources[1].cpu().numpy())
    else:
        # by copying
        shutil.copy(mix_file, mix_file2)
        shutil.copy(spk1_reverb_file, spk1_reverb_file2)
        shutil.copy(spk2_reverb_file, spk2_reverb_file2)

    # define 2 more output files (after separation)
    spk1_sep_file = os.path.join(ex_save_dir, f'{prefix}-s1.wav')
    spk2_sep_file = os.path.join(ex_save_dir, f'{prefix}-s2.wav')

    # swap rows in estimated sources if needed
    M = [[0,0], [0,0]]
    for i in range(2):
        s1 = sources[i].cpu().numpy()
        for j in range(2):
            s2 = est_sources_np[j].cpu().numpy()
            v = np.sum(np.abs(s1-s2))
            M[i][j] = v
            # print(f'i:{i}, j:{j}, {v:.2f}')

    if M[0][0] + M[1][1] > M[0][1] + M[1][0]:
        est_sources_np = est_sources_np[[1, 0], :]

    # write 2 more output files
    wav.write(spk1_sep_file, test_set.sample_rate, est_sources_np[0].cpu().numpy())
    wav.write(spk2_sep_file, test_set.sample_rate, est_sources_np[1].cpu().numpy())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file",
                        default="local/mixit_conf.yml",
                        help="Full path to save best validation model")
    parser.add_argument("--model_path",
                        default="",
                        help="model path (optional)")
    parser.add_argument("--dataset",
                        default='Echo2Mix',
                        help="dataset name")
    parser.add_argument("--sample_rate",
                        type=int,
                        default="16000",
                        help="data sample rate (override the sample rate in the config)")
    parser.add_argument("--num_outputs",
                        type=int,
                        default=0,
                        help="number of the output separated samples for checking/demo")
    parser.add_argument("--save_output",
                        type=str2bool,
                        nargs='?', # make this argument optional
                        const=True, # if the --save_output flag is present but no value is given, it defaults to True
                        default=False, # if --save_output is not provided, the default value is False
                        help="save outputs (default: False)")
    args = parser.parse_args()
    return args

compute_metrics = ["si_sdr", "sdr"]
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # adjust based on available GPU
device_id = os.environ['CUDA_VISIBLE_DEVICES']
device = torch.device(f"cuda:{device_id}")

def main(config):

    model_path = config["model_path"]
    dataset = config["dataset"]
    sample_rate = config["sample_rate"]
    num_outputs = config["num_outputs"]
    save_output = config["save_output"]

    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress), 
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )

    # import pdb; pdb.set_trace()
    # config["train_conf"]["main_args"] = {}
    config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    exp_dir = config["train_conf"]["main_args"]["exp_dir"]
    print(f'exp dir: {exp_dir}')

    # validate model path
    if len(model_path) == 0:
        # model_path = os.path.join(exp_dir, "best_model.pth")
        # model_path = os.path.join(exp_dir, "last.ckpt")
        model_path = os.path.join(exp_dir, "last.pth")
        print(f'set default model path: {model_path}')

    # roll back to the ckpt model (*.ckpt) if the clean model (*.pth) does not exist 
    ext = os.path.splitext(model_path)[-1]
    if ext == '.pth' and not os.path.isfile(model_path):
        model_path = model_path.replace('.pth', '.ckpt')
        print(f'use {model_path} (.ckpt instead of .pth)')

    assert os.path.isfile(model_path), f'model path: {model_path} does not exist!'
    print(f'model path: {model_path}')
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})

    # clean up model if it is a checkpoint model
    if os.path.splitext(model_path)[-1] == '.ckpt':

        # Define data module
        print_only("Instantiating datamodule <{}>".format(config["train_conf"]["datamodule"]["data_name"]))
        datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
            **config["train_conf"]["datamodule"]["data_config"]
        )
        datamodule.setup()

        train_loader, val_loader, test_loader = datamodule.make_loader

        # Define model and optimizer
        print_only("Instantiating AudioNet <{}>".format(config["train_conf"]["audionet"]["audionet_name"]))
        model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"])(
            sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
            **config["train_conf"]["audionet"]["audionet_config"],
        )
        print_only("Instantiating Optimizer <{}>".format(config["train_conf"]["optimizer"]["optim_name"]))
        optimizer = make_optimizer(model.parameters(), **config["train_conf"]["optimizer"])

        # Define scheduler
        scheduler = None
        if config["train_conf"]["scheduler"]["sche_name"]:
            print_only(
                "Instantiating Scheduler <{}>".format(config["train_conf"]["scheduler"]["sche_name"])
            )
            if config["train_conf"]["scheduler"]["sche_name"] != "DPTNetScheduler":
                scheduler = getattr(torch.optim.lr_scheduler, config["train_conf"]["scheduler"]["sche_name"])(
                    optimizer=optimizer, **config["train_conf"]["scheduler"]["sche_config"]
                )
            else:
                scheduler = {
                    "scheduler": getattr(look2hear.system.schedulers, config["train_conf"]["scheduler"]["sche_name"])(
                        optimizer, len(train_loader) // config["train_conf"]["datamodule"]["data_config"]["batch_size"], 64
                    ),
                    "interval": "step",
                }

        # Define Loss function.
        print_only(
            "Instantiating Loss, Train <{}>, Val <{}>".format(
                config["train_conf"]["loss"]["train"]["sdr_type"], config["train_conf"]["loss"]["val"]["sdr_type"]
            )
        )
        loss_func = {
            "train": getattr(look2hear.losses, config["train_conf"]["loss"]["train"]["loss_func"])(
                getattr(look2hear.losses, config["train_conf"]["loss"]["train"]["sdr_type"]),
                **config["train_conf"]["loss"]["train"]["config"],
            ),
            "val": getattr(look2hear.losses, config["train_conf"]["loss"]["val"]["loss_func"])(
                getattr(look2hear.losses, config["train_conf"]["loss"]["val"]["sdr_type"]),
                **config["train_conf"]["loss"]["val"]["config"],
            ),
        }

        print_only("Instantiating System <{}>".format(config["train_conf"]["training"]["system"]))
        system = getattr(look2hear.system, config["train_conf"]["training"]["system"])(
            audio_model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scheduler=scheduler,
            config=config["train_conf"],
        )

        device_id = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        state_dict = torch.load(model_path, map_location=f'cuda:{device_id}')
        system.load_state_dict(state_dict=state_dict["state_dict"])
        system.cpu()

        to_save = system.audio_model.serialize()
        clean_model_path = model_path.replace('.ckpt', '.pth')
        epoch = state_dict['epoch']
        model_name = f'ep{epoch}'
        clean_model_path = clean_model_path.replace('last', model_name)
        torch.save(to_save, os.path.join(exp_dir, clean_model_path))

        print(f'coverted model from {model_path} to {clean_model_path}')
        model_path = clean_model_path

    # config["model_name"] = "SPMamba"
    model =  getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
        model_path,
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )

    if config["train_conf"]["training"]["gpus"]:
        if not device:
            device = "cuda"
        model.to(device)
    model_device = next(model.parameters()).device

    # datamodule_ori: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
    #     **config["train_conf"]["datamodule"]["data_config"])
    # datamodule_ori.setup()
    # _, _ , test_set_ori = datamodule_ori.make_sets

    sr_type = {8000: '8k', 16000: '16k'}
    sr = sr_type[sample_rate]
    # config["train_conf"]["datamodule"]["data_config"]["segment"] = 4.0
    config["train_conf"]["datamodule"]["data_config"]["train_dir"] = f"data/{dataset}/{sr}/train"
    config["train_conf"]["datamodule"]["data_config"]["valid_dir"] = f"data/{dataset}/{sr}/val"
    config["train_conf"]["datamodule"]["data_config"]["test_dir"] = f"data/{dataset}/{sr}/test"

    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"])
    datamodule.setup()
    _, _ , test_set = datamodule.make_sets
   
    # set the output dir
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    ex_save_dir = os.path.join(exp_dir, "results", model_name, f'{dataset}-{sr}')
    if os.path.isdir(ex_save_dir):
        print(f'use existing output dir: {ex_save_dir}')
    else:
        os.makedirs(ex_save_dir)
        print(f'create new output dir: {ex_save_dir}')

    # Randomly choose the indexes of sentences to save.
    csvfile = os.path.join(ex_save_dir, "metrics.csv")
    metrics = MetricsTracker(save_file=csvfile)
    torch.no_grad().__enter__()

    # get num. of testing samples
    num_test_set = len(test_set)
    print(f'# testing samples: {num_test_set}')
    # ndigits = len(str(num_test_set))

    # get set-wise indeces for combined dataset
    idxs1, idxs2 = [], []
    if dataset == 'WSJ0-Echo-2Mix':
        for idx in range(num_test_set):
            basename = os.path.basename(test_set.mix[idx][0])
            if basename == 'mix.wav':
                idxs2.append(idx)
            else:
                idxs1.append(idx)

    # evaluate on the selected samples to get speech output
    for i in range(num_outputs):
        for idxs in [idxs1, idxs2]:
            idx = idxs[i]

            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx], device=model_device)
            print(f'i:{i}, idx:{idx}, key:{key}')
            est_sources = model(mix[None]) # #samples X n_src X sample length
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0) # n_src X sample length

            write_outputs(test_set, idx, est_sources_np, ex_save_dir, dataset=None)

    # evaluate on all samples to get the avg. metrics
    with progress:
        for idx in progress.track(range(num_test_set)):

            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(test_set[idx],
                                                    device=model_device)
            est_sources = model(mix[None]) # #samples X n_src X sample length
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0) # n_src X sample length

            metrics(mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np,
                    key=key)
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
    metrics.final()

    # backup csv file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csvfile2 = csvfile.replace('.csv', f'_{timestamp}.csv')
    shutil.copy(csvfile, csvfile2)

if __name__ == "__main__":

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    ## WSJ0-2Mix
    # args.dataset = 'WSJ0-2Mix' # WSJ0-2Mix, Echo2Mix, WSJ0-Echo-2Mix
    # args.conf_file = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", "conf.yml")
    # args.model_path = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'last.pth')
    # args.model_path = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'ep153.pth')
    # args.model_path = os.path.join(os.getcwd(), 'experiments', 'checkpoint', f"SPMamba-{args.dataset}", 'best_model.pth')

    # # WSJ0-Echo-2Mix_norm_std
    # args.dataset = 'WSJ0-Echo-2Mix_norm_std' # WSJ0-2Mix, Echo2Mix, WSJ0-Echo-2Mix
    # args.conf_file = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'conf.yml')
    # args.model_path = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'best_model.pth')

    # # WSJ0-Echo-2Mix
    # args.dataset = 'WSJ0-Echo-2Mix' # WSJ0-2Mix, Echo2Mix, WSJ0-Echo-2Mix
    # args.conf_file = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'conf.yml')
    # args.model_path = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'last.ckpt')
    # args.model_path = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'ep64.pth')
    # args.model_path = os.path.join(os.getcwd(), "experiments", "checkpoint", f"SPMamba-{args.dataset}", 'best_model.pth')

    # args.sample_rate = 16000
    # args.num_outputs = 10
    # args.save_output = True

    # check file/dir existence
    assert os.path.isfile(args.conf_file), f"config file: {args.conf_file} does not exist!"
    assert os.path.isfile(args.model_path), f"model file: {args.model_path} does not exist!"

    # print input arguments
    print(f'conf dir: {args.conf_file}')
    print(f'model path: {args.model_path}')
    print(f'data set: {args.dataset}')
    print(f'sample rate: {args.sample_rate}')
    print(f'num of outputs: {args.num_outputs}')
    print(f'save output: {args.save_output}')

    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_file, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)

    config = arg_dic
    main(config)
