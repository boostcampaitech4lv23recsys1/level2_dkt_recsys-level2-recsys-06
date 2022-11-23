import os
import argparse

from trainer import Trainer
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="[LG U+] AI GROUND"
    )
    parser.add_argument("--data_path", type = str, default = "/opt/ml/input/data/")
    parser.add_argument("--output_path", type = str, default = "/opt/ml/output")
    
    parser.add_argument("--valid_mode", type = str, default = 'kfold', help = "kfold, random")
    parser.add_argument("--iterations", type = int, default = 100)
    parser.add_argument("--learning_rate", type = float, default = 3e-1)
    parser.add_argument("--depth", type = int, default = 10)
    parser.add_argument("--eval_metric", type = str, default = "AUC")

    return parser.parse_args()


def main():
    args = parse_args()
    params = {'iterations': args.iterations, 'learning_rate': args.learning_rate, 'depth': args.depth, 'eval_metric': args.eval_metric}

    trainer = Trainer(args, params)

    print("start train")
    submission = trainer.training()

    opath = Path(os.path.join(args.output_path, "submission.csv"))
    opath.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(str(opath), index = False)
    
if __name__ == '__main__':
    main()