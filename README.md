# ADL22-HW2

## Context Selection
```shell
python context_selection.py [--mode mode] [--ckpt_dir dir] [--context_path path] [--input_file file] [--val_file file] [--output_path path] [--max_len len] [--pretrained_model] [--lr lr] [--gradient_accumulation step] [--num_epoch epoch] [--device device] [--logging_step step]
```
- `--mode`: train|test. Default="train"
- `--ckpt_dir`: Directory to save the model file. Default="./ckpt/context_selection/"
- `--context_path`: Path to the context file. Default="./data/context.json"
- `--input_file`: Path to the training or testing file. Default="./data/train.json"
- `--val_file`: Path to the validation file. If this argument is provided, the validation data will be added to the training set. This argument will only be used when `mode=train`.
- `--output_path`: Path of the output file. This argument will only be used when `mode=test`. Default="./selection.json"
- `--max_len`: Default=512
- `--pretrained_model`: Pretrained model from [Hugging Face](https://huggingface.co/models). Default="bert-base-chinese"
    - `mode=train`: The finetuned model will be save under `ckpt_dir/pretrained_model/`. 
    - `mode=test`: Read the configuration from `ckpt_dir/pretrained_model/`.
- `--lr`: Default=5e-5
- `--gradient_accumulation`: Number of accumulation steps before updating the model parameters. This argument will only be used when `mode=train`. Default=64 $$Effective\ batch size = batch\ size * gradient\ accumulation\ steps$$
- `--num_epoch`: Number of training epoch. This argument will only be used when `mode=train`. Default=1
- `--device`: cpu, cuda, cuda:0, cuda:1. Default="cuda"
- `--logging_step`: Number of steps before outputing training log. This argument will only be used when `mode=train`. Default=100

## Converting the output data of our context selection model to the [SQuAD](https://huggingface.co/datasets/squad) format
```shell
python to_SQuAD.py [--context path] <--input_file file> [--output_file file]
```
- `--context`: Path to the context file. Default="./data/context.json"
- `--input_file`: Path to the input file. 
- `--output_file`: Path to the output file. Default="./squad_like.json"

## Question Answering
Please refer to the [original github link](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) for more information. I will only show the parameters that I have used for running this script.
```shell
python question_answering.py <--model_name_or_path model> [<--do_train> <--train_file file>] [<--do_eval> <--validation_file file> [--evaluation_strategy strategy] [--eval_steps step]] [<--do_predict> <--test_file file>] [--output_dir dir] 
```
- `--model_name_or_path`: Path to pretrained model or model identifier from huggingface.co/models.
- `--do_train`: Whether to run training. (default: False)
- `--train_file`: The input training data file (a text file).
- `--do_eval`: Whether to run eval on the dev set. (default: False)
- `--validation_file`: An optional input evaluation data file to evaluate the perplexity on (a text file).
- `--eval_steps`: Run an evaluation every X steps. (default: None)
- `--do_predict`: Whether to run predictions on the test set. (default: False)
- `--test_file` An optional input test data file to evaluate the perplexity on (a text file).
- `--evaluation_strategy`: {no,steps,epoch} The evaluation strategy to use. (default: no)
- `--output_dir`: The output directory where the model predictions and checkpoints will be written. (default: None)

## Converting the output data of our question selection model to the required submission format 
```shell
python to_submission.py [--input_file file] [--output_file file]
```
- `--input_file`: Path to the input file. Default="./predict_predictions.json"
- `--output_file`: Path to the output file. Default="./submission.csv"


## Intent Classification (BONUS)
```
python intent_classification.py [--mode mode] [--ckpt_dir dir] [--input_file file] [--mapping map] [--output_path path] [--max_len len] [--pretrained_model] [--lr lr] [--gradient_accumulation step] [--num_epoch epoch] [--device device] [--logging_step step]
```
- `--mode`: train|test. Default="train"
- `--ckpt_dir`: Directory to save the model file. Default="./ckpt/intent_classification/"
- `--input_file`: Path to the training or testing file. Default="../ADL21-HW1/data/intent/train.json"
- `--mapping`: File that store a mapping between intent and index. Default="../ADL21-HW1/cache/intent/intent2idx.json"
- `--output_path`: Path of the output file. This argument will only be used when `mode=test`. Default="./intent.json"
- `--max_len`: Default=512
- `--pretrained_model`: Pretrained model from [Hugging Face](https://huggingface.co/models). Default="bert-base-uncased"
    - `mode=train`: The finetuned model will be save under `ckpt_dir/pretrained_model/`. 
    - `mode=test`: Read the configuration from `ckpt_dir/pretrained_model/`.
- `--lr`: Default=5e-5
- `--gradient_accumulation`: Number of accumulation steps before updating the model parameters. This argument will only be used when `mode=train`. Default=64 $$Effective\ batch size = batch\ size * gradient\ accumulation\ steps$$
- `--num_epoch`: Number of training epoch. This argument will only be used when `mode=train`. Default=2
- `--device`: cpu, cuda, cuda:0, cuda:1. Default="cuda"
- `--logging_step`: Number of steps before outputing training log. This argument will only be used when `mode=train`. Default=500
