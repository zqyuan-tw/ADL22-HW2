python context_selection.py --mode test --context_path "${1}" --input_file "${2}" --pretrained_model hfl/chinese-bert-wwm-ext 
python to_SQuAD.py --input_file selection.json
python question_answering.py --model_name_or_path ckpt/question_answering/hfl/chinese-roberta-wwm-ext --test_file squad_like.json --output_dir . --do_predict
python to_submission.py --output_file "${3}"
