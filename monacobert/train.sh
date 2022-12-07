# models="monacobert_ctt bert_ctt monabert_ctt cobert_ctt"
# dataset_names="icecream_pid_diff"

# models="monacobert"
# dataset_names="icecream_pid"

# best
models="monacobert_ctt"
dataset_names="icecream_pid_diff"

for model in ${models}
do
    for dataset_name in ${dataset_names}
    do
        python train.py \
        --model_fn ${model}_${dataset_name}.pth \
        --model_name ${model} \
        --dataset_name ${dataset_name} \
        --batch_size 128 \
        --max_seq_len 32 \
        --n_epochs 1000
    done
done