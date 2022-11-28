import time
import argparse
import pandas as pd

from src.utils import seed_everything

from src.data_load.boost_data import boost_data_load
from src.models.boost_models import LightGBM, CatBoost



def main(args):
    seed_everything(args.SEED)

    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('LGBM', 'CATBOOST'):
        # csv로부터 data를 불러온 후 피쳐 엔지니어링을 하고 train/test를 분리해서 전달
        data = boost_data_load(args)
    else:
        pass

    ######################## Train/Valid Split
    # print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    # if args.MODEL in ('LGBM', 'CATBOOST'):
    #     data = context_data_split(args, data)
    #     data = context_data_loader(args, data)
    # else:
    #     pass

    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='LGBM':
        model = LightGBM(args, data)
    elif args.MODEL=='CATBOOST':
        model = CatBoost(args, data)
    else:
        pass

    ######################## TRAIN and Inference
    print(f'--------------- {args.MODEL} TRAINING and INFERENCING ---------------')
    predicts = model.train()


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ('LGBM', 'CATBOOST'):
        submission['prediction'] = predicts
    else:
        pass

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='opt/ml/input/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, choices=['LGBM', 'CATBOOST'],
                                help='LGBM, CATBOOST 중에서 선택하세요.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--CV', type=str, default='kfold', help='교차검증 방법을 선택하세요. 기본값: kfold')
    arg('--N_SPLITS', type=int, default=5, help='K-Fold 교차검증 시 사용할 스플릿 값을 지정하세요.')
    
    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')
    
    ############### LIGHT GBM
    arg('--NUM_BOOST_ROUND', type=int, default=500, help='LGBM의 부스트 라운드 횟수를 정하세요.')
    arg('--VERBOSE_EVAL', type=int, default=100, help='LGBM 학습 중 현황이 나오는 주기를 정하세요.')
    arg('--EARLY_STOPPING_ROUND', type=int, default=100, help='얼리 스타핑을 실행하는 라운드를 정하세요.')

    ############### CATBOOST
    arg('--VERBOSE', type=int, default=0, help='학습 중 설명을 들을지의 여부를 Yes는 1, No는 0으로 적어주세요.')



    args = parser.parse_args()
    main(args)
