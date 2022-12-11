import os
import pickle

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds



def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 텍스트 전처리
    preprocess = Preprocess(args)

    # 훈련 데이터 불러오기. 이 때 feature engineering까지 완료된다
    preprocess.load_data(args.file_name)

    # 불러온 훈련 데이터 저장하기
    train_data = preprocess.get_train_data()

    # 저장한 훈련 데이터 valid_data로 나누기
    train_data, valid_data = preprocess.split_data(train_data)

    wandb.init(project="dkt", config=vars(args))

    # 모델 인스턴스 생성
    model = trainer.get_model(args).to(args.device)

    # 생성한 인스턴스로 훈련 시키기
    # get_loader로 loader를 불러와 훈련시작
    trainer.run(args, train_data, valid_data, model)

    test_data = preprocess.get_test_data()

    model = trainer.load_model(args).to(args.device)
    
    trainer.inference(args, test_data, model)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
