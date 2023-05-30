# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset, concatenate_datasets
from loguru import logger

sys.path.append('../..')
from textgen import LlamaModel


def load_data(train_dataset_dir):
    path = Path(train_dataset_dir)
    files = [os.path.join(path, file.name) for file in path.glob("*.json")]
    logger.info(f"training files: {' '.join(files)}")
    all_datasets = []
    for file in files:
        raw_dataset = load_dataset("json", data_files=file)
        all_datasets.append(raw_dataset["train"])
    all_datasets = concatenate_datasets(all_datasets)
    data = []
    logger.debug(f"all_datasets size:{len(all_datasets)}, first line: {next(iter(all_datasets))}")
    for example in all_datasets:
        instruction, intput_str, output_str = example["instruction"], example["input"], example["output"]
        instruction = instruction + intput_str
        data.append([instruction, '', output_str])
    logger.info(f"total data size:{len(data)}, head: {data[:3]}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='../data/json_files/', type=str, help='Training data file')
    parser.add_argument('--test_data_dir', default='../data/json_files/', type=str, help='Test data file')
    parser.add_argument('--model_type', default='llama', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='shibing624/chinese-alpaca-plus-7b-hf', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--is_train_on_prompt', action='store_true', help='Whether to compute loss on prompt')
    parser.add_argument('--output_dir', default='./outputs-mydata/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=256, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=256, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=0.2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--eval_steps', default=50, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=50, type=int, help='Save checkpoint every X steps')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune Llama model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "use_peft": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "is_train_on_prompt": args.is_train_on_prompt,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
        }
        model = LlamaModel(args.model_type, args.model_name, args=model_args)
        train_data = load_data(args.train_data_dir)
        logger.debug('train_data: {}'.format(train_data[:3]))
        train_df = pd.DataFrame(train_data, columns=["instruction", "input", "output"])
        eval_df = train_df[:10]
        train_df = train_df[10:]
        model.train_model(train_df, eval_data=eval_df)
    if args.do_predict:
        if model is None:
            model = LlamaModel(
                args.model_type, args.model_name,
                args={'use_peft': True, 'eval_batch_size': args.batch_size,
                      'output_dir': args.output_dir, "max_length": args.max_length, }
            )
        test_data = load_data(args.test_data_dir)[:10]
        test_df = pd.DataFrame(test_data, columns=["instruction", "input", "output"])
        logger.debug('test_df: {}'.format(test_df))

        def get_prompt(arr):
            if arr['input'].strip():
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{arr['instruction']}\n### Input:\n{arr['input']}\n\n### Response:\n\n"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{arr['instruction']}\n\n### Response:\n\n"""

        test_df['prompt'] = test_df.apply(get_prompt, axis=1)
        test_df['predict_after'] = model.predict(test_df['prompt'].tolist())
        logger.debug('test_df result: {}'.format(test_df[['output', 'predict_after']]))
        out_df = test_df[['instruction', 'input', 'output', 'predict_after']]
        out_df.to_json('test_result.json', force_ascii=False, orient='records', lines=True)


if __name__ == '__main__':
    main()
