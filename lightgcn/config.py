import time

# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    # wandb_kwargs = dict(project="dkt-gcn")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    #preprocessing
    valid_num = 1

    # dump
    output_dir = "./output/"
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    pred_file = "submission_{}.csv".format(save_time)

    # build
    embedding_dim = 512  # int
    num_layers = 1  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    n_epoch = 3000
    learning_rate = 0.001
    weight_basepath = "./weight"


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
