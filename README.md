# MatchSum
Code for ACL 2020 paper: *[Extractive Summarization as Text Matching](https://arxiv.org/abs/2004.08795)*


## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.4.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 0.5.0
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
	- You should fill your ROUGE path in metrics.py line 20 before running our code.
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
	- Used in  the validation phase.
- [transformers](https://github.com/huggingface/transformers) 2.5.1

	
All code only supports running on Linux.

## Data

We have already processed CNN/DailyMail dataset, you can download it through [this link](https://drive.google.com/open?id=1FG4oiQ6rknIeL2WLtXD0GWyh6pBH9-hX), unzip and move it to `./data`. It contains two versions (BERT/RoBERTa) of the dataset, a total of six files.

In addition, we have released five other processed datasets (WikiHow, PubMed, XSum, MultiNews, Reddit), which you can find [here](https://drive.google.com/file/d/1PnFCwqSzAUr78uEcA_Q15yupZ5bTAQIb/view?usp=sharing).

## Train

We use eight Tesla-V100-16G GPUs to tra1in our model, the training time is about 30 hours. If you do not have enough video memory, you can reduce the *batch_size* or *candidate_num* in `train_matching.py`, or you can adjust *max_len* in `dataloader.py`.

You can choose BERT or RoBERTa as the encoder of **MatchSum**,  for example, to train a RoBERTa model, you can run the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_matching.py --mode=train --encoder=roberta --save_path=./roberta --gpus=0,1,2,3,4,5,6,7
```

## Test

After completing the training process, several best checkpoints are stored in a folder named after the training start time, for example, `./roberta/2020-04-12-09-24-51`. You can run the following command to get the results on test set (only one GPU is required for testing):

```
CUDA_VISIBLE_DEVICES=0 python train_matching.py --mode=test --encoder=roberta --save_path=./roberta/2020-04-12-09-24-51/ --gpus=0
```
The ROUGE score will be printed on the screen, and the output of the model will be stored in the folder  `./roberta/result`.

## Results on CNN/DailyMail
Test set (the average of three runs)

| Model | R-1 | R-2 | R-L |
| :------ | :------: | :------: | :------: |
| MatchSum (BERT-base) | 44.22 | 20.62 | 40.38 | 
| MatchSum (RoBERTa-base) | 44.41 | 20.86 | 40.55 |

## Generated Summaries
The summaries generated by our models on the CNN/DM dataset can be found [here](https://drive.google.com/open?id=11_eSZkuwtK4bJa_L3z2eblz4iwRXOLzU). In the version we released, the result of **MatchSum(BERT)** is 44.26/20.58/40.40 (R-1/R-2/R-L), and the result of **MatchSum(RoBERTa)** is 44.45/20.88/40.60.

The summaries generated on other datasets can be found [here](https://drive.google.com/open?id=1iNY1hT_4ZFJZVeyyP1eeoVY14Ej7l9im).

## Pretrained Model
Two versions of the pre-trained model on CNN/DM are available [here](https://drive.google.com/file/d/1PxMHpDSvP1OJfj1et4ToklevQzcPr-HQ/view?usp=sharing). You can use them through `torch.load`. For example,

```
model = torch.load('MatchSum_cnndm_bert.ckpt')
```

Besides, the pre-trained models on other datasets can be found [here](https://drive.google.com/open?id=1EzRE7aEsyBKCeXJHKSunaR89QoPhdij5).

## Process Your Own Data

If you want to process your own data and get candidate summaries for each document, first you need to convert your dataset to the same *jsonl* format as ours, and make sure to include *text* and *summary* fields. Second, you should use BertExt or other methods to select some important sentences from each document and get an *index.jsonl* file (we provide an example in `./preprocess/test_cnndm.jsonl`).

Then you can run the following command:

```
python get_candidate.py --tokenizer=bert --data_path=/path/to/your_original_data.jsonl --index_path=/path/to/your_index.jsonl --write_path=/path/to/store/your_processed_data.jsonl
```

Please fill your ROUGE path in `preprocess/get_candidate.py` line 22 before running this command. It is worth noting that you need to adjust the number of candidate summaries and the number of sentences in the candidate summaries according to your dataset. For details, see line 89-97 in `preprocess/get_candidate.py`.

After processing the dataset, and before using our code to train your own model, please adjust *candidate_num* in `train_matching.py` and *max_len* in `dataloader.py` according to the number and the length of the candidate summaries in your dataset.

## Note

The code and data released here are used for the matching model. Before the matching stage, we use BertExt to prune meaningless candidate summaries, the implementation of BertExt can refer to [PreSumm](https://github.com/nlpyang/PreSumm).
