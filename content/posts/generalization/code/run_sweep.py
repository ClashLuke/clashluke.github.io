import argparse
import csv
import hashlib
import io
import json
import os
import random
import traceback

import torch
import torch.multiprocessing as mp

from data import get_tensors
from train import LAYER_NAMES, NORM_FAMILIES, train_one

DATASETS = ['cifar10', 'cifar100']
IPC_C10 = [1, 4, 16, 64, 256, 1024]
IPC_C100 = [1, 4, 16, 64, 256]
MAX_IPC = {'cifar10': 1024, 'cifar100': 256}
WIDTHS = (4, 8, 16, 32)

CONFIG_FIELDS = [
    'config_id', 'dataset', 'images_per_class', 'batch_size',
    'width', 'use_bn', 'lr', 'weight_decay', 'dropout',
    'rho', 'asam_rho', 'label_smoothing', 'ortho', 'seed',
]
SUMMARY_FIELDS = [
    'peak_test_acc', 'train_acc', 'steps', 'wall_time', 'event',
    'test_loss', 'train_loss', 'param_dist',
]
LAYER_FIELDS = [f'{fam}_{layer}' for fam in NORM_FAMILIES
                for layer in LAYER_NAMES + ['global']]
FIELDS = CONFIG_FIELDS + SUMMARY_FIELDS + LAYER_FIELDS
KEY_FIELDS = CONFIG_FIELDS[1:]


def sample_config(rng, seed):
    ds = rng.choice(DATASETS)
    ipc = rng.choice([i for i in (IPC_C10 if ds == 'cifar10' else IPC_C100)])
    num_classes = 10 if ds == 'cifar10' else 100
    bs_choices = [b for b in [4, 16, 64, 256] if b <= ipc * num_classes]
    return dict(
        dataset=ds, images_per_class=ipc,
        batch_size=rng.choice(bs_choices),
        width=rng.choice(WIDTHS),
        use_bn=rng.random() > 0.5,
        lr=10 ** (rng.random() * 3 - 4),
        weight_decay=rng.random() * 0.1,
        dropout=rng.random() * 0.7,
        rho=rng.random() * 0.3 if rng.random() > 0.5 else 0.0,
        asam_rho=rng.random() * 0.1 if rng.random() > 0.5 else 0.0,
        label_smoothing=rng.random() * 0.3,
        ortho=rng.random() > 0.5,
        seed=seed,
    )


def cfg_key(c):
    return hashlib.sha256(
        json.dumps({k: c[k] for k in KEY_FIELDS}, sort_keys=True).encode()
    ).hexdigest()[:16]


def load_done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        lines = f.readlines()
    if lines and not lines[-1].endswith('\n'):
        lines = lines[:-1]
    return {row['config_id'] for row in csv.DictReader(io.StringIO(''.join(lines)))}


def load_done_all(out_dir):
    if not os.path.isdir(out_dir):
        return set()
    done = set()
    for name in os.listdir(out_dir):
        if name.endswith('.csv'):
            done |= load_done(os.path.join(out_dir, name))
    return done


def worker(gpu_id, worker_idx, out_dir, data_dir, seed):
    torch.cuda.set_device(gpu_id)
    csv_path = os.path.join(out_dir, f'gpu{gpu_id}_w{worker_idx}.csv')
    trace_dir = os.path.join(out_dir, 'traces')
    os.makedirs(trace_dir, exist_ok=True)

    has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if not has_header:
        writer.writeheader()
        csv_file.flush()

    done = load_done_all(out_dir)
    rng = random.Random()
    ds_cache, data, count = None, None, 0
    device = f'cuda:{gpu_id}'

    try:
        while True:
            cfg = sample_config(rng, seed)
            cid = cfg_key(cfg)
            if cid in done:
                continue
            done.add(cid)

            ds_key = (cfg['dataset'], cfg['images_per_class'])
            if ds_key != ds_cache:
                data = tuple(t.to(device) for t in
                             get_tensors(cfg['dataset'], cfg['images_per_class'], data_dir))
                ds_cache = ds_key

            try:
                summary, trace = train_one(cfg, *data)
            except KeyboardInterrupt:
                raise
            except Exception:
                torch.cuda.empty_cache()
                summary = {f: 0 for f in SUMMARY_FIELDS + LAYER_FIELDS}
                summary['event'] = 'error'
                trace = None
                traceback.print_exc()

            writer.writerow({**cfg, 'config_id': cid, **summary})
            csv_file.flush()
            if trace is not None:
                with open(os.path.join(trace_dir, f'{cid}.json'), 'w') as tf:
                    json.dump(trace, tf)
            count += 1
            if count % 10 == 0:
                print(f'[GPU {gpu_id}] {count} done', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, required=True)
    parser.add_argument('--seed', type=int, default=161)
    parser.add_argument('--out-dir', type=str, default='./sweep_results')
    parser.add_argument('--data-dir', type=str, default='./data')
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)
    done = load_done_all(args.out_dir)
    print(f'{len(done)} done, {len(gpus)} GPUs')

    processes = []
    for gpu_id in gpus:
        p = mp.Process(target=worker, args=(gpu_id, 0, args.out_dir, args.data_dir, args.seed))
        p.start()
        processes.append(p)
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
