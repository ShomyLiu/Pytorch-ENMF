## ENMF

This repo reproduces the paper with Pytorch:

>Chong Chen, Min Zhang, Yongfeng Zhang, Yiqun Liu and Shaoping Ma. 2020. Efficient Neural Matrix Factorization without Sampling for Recommendation. In TOIS Vol. 38, No. 2, Article 14.

According to their origin tensorflow-based implementation: [ENMF](https://github.com/chenchongthu/ENMF)(thanks for the nice work), we implement the user-based ENMF (denoted as ENMF-U in the paper).

### Environment
- python 3.6+
- pytorch 1.4
- fire

### run

- First pre-processe the dataset:
```
cd pro_data
python3 data_pro.py
```
- Run the main for training and testing:
```
python3 main.py train
```
- an output sample:
```
2020-07-15 14:03:26 Epoch 93:
        loss: -123.066313
        Recall@50: 0.295530, ndcg@50: 0.088255
        Recall@100: 0.441887, ndcg@100: 0.111960
        Recall@200: 0.602483, ndcg@200: 0.134411
2020-07-15 14:03:29 Epoch 94:
        loss: -123.118651
        Recall@50: 0.298179, ndcg@50: 0.088913
        Recall@100: 0.439404, ndcg@100: 0.111792
        Recall@200: 0.603311, ndcg@200: 0.134731
2020-07-15 14:03:31 Epoch 95:
        loss: -123.206238
        Recall@50: 0.297682, ndcg@50: 0.088444
        Recall@100: 0.440563, ndcg@100: 0.111570
        Recall@200: 0.603311, ndcg@200: 0.134322

```

Some experiment settings are in `config.py`.
