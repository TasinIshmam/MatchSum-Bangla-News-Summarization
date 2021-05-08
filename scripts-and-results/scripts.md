

#### Preprocessing

- Run preprocessing (test)   
```shell  
python get_candidate.py --tokenizer=bert --data_path=./testing-preprocess/eg-data.jsonl --index_path=./testing-preprocess/eg-index.jsonl  --write_path=./testing-preprocess/preprocess-output.jsonl
```

- Run preprocessing (bangla example)
```shell
python get_candidate.py --tokenizer=bert --data_path=./data-bangla/data-eg-matchsum-format.jsonl --index_path=./data-bangla/sentid-eg-matchsum-format.jsonl  --write_path=./testing-preprocess/preprocess-output-bangla-eg.jsonl
```
- Run preprocessing (Bangla all) 
```shell
python get_candidate.py --tokenizer=bert --data_path=./data-bangla/data-all-array-of-strings-matchsum-format.jsonl --index_path=./data-bangla/sentid-all-matchsum-format.jsonl  --write_path=./data-bangla/preprocess-output-bangla-all.jsonl
```


#### Validation/Test 

- Using BERT on CNN/DM dataset (reads model weights from ./models and outputs to ./result)  
    - Note that there is a slight typo in the original readme, which says that the output will be stored to ./models/results.
```sh
CUDA_VISIBLE_DEVICES=0 python train_matching.py --mode=test --encoder=bert --save_path=./models --gpus=0
```


#### Train
- Using BERT on CNN/DM dataset. (Reads data from ./data and outputs to ./bert-output)
```sh
CUDA_VISIBLE_DEVICES=0 python train_matching.py --mode=train --encoder=bert --save_path=./bert-output --gpus=0
```