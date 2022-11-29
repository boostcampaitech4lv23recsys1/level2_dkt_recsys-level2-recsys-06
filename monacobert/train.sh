model="monacobert_ctt"
dataset_name="icecream_pid_diff"

python train.py \
--model_fn ${model}_${dataset_name}.pth \
--model_name ${model} \
--dataset_name ${dataset_name} \
--batch_size 128 \
--grad_acc True \
--grad_acc_iter 4 \
--max_seq_len 128 \
--n_epochs 100 