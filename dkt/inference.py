import os
import numpy as np
import pickle
import torch
from args import parse_args
from src import trainer
from src.dataloader import Preprocess


def main(args):
    total_preds = np.zeros((5, 744))
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)
    with open(os.path.join(args.asset_dir, "preprocess.pckl"), "rb") as f:
        preprocess.duration_normalizer = pickle.load(f)
    preprocess.load_test_data(args.test_file_name)

    test_data = preprocess.get_test_data()

    if args.group_mode == 'userid_with_testid':
        _, testset = preprocess.concat_for_train(test_data)
    else:
        testset = test_data

    if args.valid_mode == 'kfold':
        for idx in range(5):
            fold_num = idx + 1
            model = trainer.load_model(args, fold_num).to(args.device)
            pred = trainer.inference(args, testset, model)
            pred = np.array(pred)

            total_preds[idx] = pred
        total_preds = np.mean(total_preds, axis = 0).tolist()

    else:
        model = trainer.load_model(args, 0).to(args.device)
        pred = trainer.inference(args, testset, model)    
        total_preds = pred
        
    write_path = os.path.join(args.output_dir, "submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
