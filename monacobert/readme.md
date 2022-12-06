# How to run this code?

If you want to run the MonaCoBERT, you have to use pid_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert --dataset_name icecream_pid
```

If you want to run the MonaCoBERT_CTT, you have to use pid_diff_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert_ctt --dataset_name icecream_pid_diff
```