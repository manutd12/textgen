# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

import pandas as pd
from loguru import logger

sys.path.append('../..')
from textgen import LlamaModel


def load_data(file_path):
    return [i for i in open(file_path, 'r', encoding='utf-8').read().split('\n\n') if i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/pt.txt', type=str, help='Training data file')
    parser.add_argument('--test_file', default='../data/pt.txt', type=str, help='Test data file')
    parser.add_argument('--model_type', default='llama', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='shibing624/chinese-llama-plus-13b-hf', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./outputs-pretraining/', type=str, help='Model output directory')
    parser.add_argument('--block_size', default=1024, type=int, help='Block size for training')
    parser.add_argument('--num_epochs', default=2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--eval_steps', default=50, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=50, type=int, help='Save checkpoint every X steps')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune Llama model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "is_pretraining": True,
            "block_size": args.block_size,
            "use_peft": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
        }
        model = LlamaModel(args.model_type, args.model_name, args=model_args)
        train_data = load_data(args.train_file)
        logger.debug(f'train_data, size: {len(train_data)}, head top3: {train_data[:3]}')
        train_df = pd.DataFrame(train_data, columns=["text"])
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
        response = model.predict(["给出三个保持健康的秘诀。"])
        print(response)
        response = model.predict(["张某某犯挪用资金罪和伪造、变造国家机关公文罪，如何处罚？"])
        print(response)


if __name__ == '__main__':
    main()
