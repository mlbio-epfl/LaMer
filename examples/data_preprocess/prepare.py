import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default='~/data/verl-agent/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_size', default=256, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)

    args = parser.parse_args()
    print(f"processing data for mode: {args.mode}")
    args.local_dir = os.path.join(args.local_dir, args.mode)

    data_source = 'hiyouga/geometry3k'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train'].select(range(args.train_data_size))
    test_dataset = dataset['test'].select(range(args.val_data_size))

    instruction_following = {
        "visual": "<image>",
        "text": "",
        }

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = instruction_following[args.mode]
            # answer = example.pop('answer')
            images = example.pop('images')

            if args.mode == 'visual':
                data = {
                    "data_source": args.mode,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "images": images,
                    "ability": "agent",
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            else:
                data = {
                    "data_source": args.mode,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "agent",
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
