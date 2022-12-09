import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--valid_mode", default="random", type=str, help="random, kfold")
    parser.add_argument("--group_mode", default="userid", type=str, help="userid, userid_with_testid")
    parser.add_argument("--computing_loss", default="custom2", type=str, help="all, last, custom, custom2")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")
    parser.add_argument("--model", default="gru_lastquery", type=str, help="lstm, lstmattn, bert, lastqt, gru_lastquery")

    parser.add_argument("--data_dir", default="/opt/ml/input/data", type=str, help="data directory")
    parser.add_argument("--asset_dir", default="/opt/ml/output/asset/", type=str, help="data directory")
    parser.add_argument("--train_file_name", default="train_data.csv", type=str, help="train file name")

    parser.add_argument("--model_dir", default="/opt/ml/output/weight/", type=str, help="model directory")
    parser.add_argument("--model_name", default="model.pt", type=str, help="model file name")

    parser.add_argument("--test_file_name", default="test_data.csv", type=str, help="test file name")

    parser.add_argument(
        "--output_dir", default="/opt/ml/input/code/dkt/output/", type=str, help="output directory"
    )

    
    parser.add_argument("--n_layers", default = 2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default = 2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")
    parser.add_argument("--max_seq_len", default=1000, type=int, help="max sequence length")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")

    # 모델
    parser.add_argument("--embed_dim", default=20, type=int, help="hidden dimension size")
    parser.add_argument("--hidden_dim", default=128, type=int, help="hidden dimension size")

    #last query transformer
    parser.add_argument("--use_lstm", default = False, type=bool, help = "last-query-transformer use lstm")

    # 훈련
    parser.add_argument("--n_epochs", default=30, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=1, type=int, help="clip grad")
    parser.add_argument("--patience", default=3, type=int, help="for early stopping")
    parser.add_argument("--wd", default=1e-2, type=float, help="weight decay")
    parser.add_argument( "--log_steps", default=50, type=int, help="print log per n steps")

    ### 중요 ###
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument("--scheduler", default="linear_warmup", type=str, help="scheduler type")

    #  # T-Fixup
    # parser.add_argument('—-Tfixup', default=False, type=bool, help='Using T-Fixup')
    # parser.add_argument('—-layer_norm', default=False, type=bool, help='T-Fixup with layer norm')
    # parser.add_argument('—-dim_div', default=3, type=int, help='model에서 dimension이 커지는 것을 방지')

    ## data agumentation
    parser.add_argument('--window', default=False, type=bool, help='Using agumentation')
    parser.add_argument('--stride', default=2, type=int, help='window stride')
    parser.add_argument('--shuffle', default=False, type=bool, help='Using shuffle')
    parser.add_argument('--shuffle_n', default=2, type=int, help='shuffle_n')
    

    args = parser.parse_args()

    return args
