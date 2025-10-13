import subprocess
import itertools
from collections import defaultdict
import wandb
from dotenv import load_dotenv
import os
import argparse


load_dotenv()
wandb.login(key = os.environ.get("WANDB_API_KEY"), relogin=True)


def main(args):
    api = wandb.Api()
    project_path = args.wandb_proj
    _, project_title = project_path.split('/', 1)

    project_exists = False
    # Figure out once whether the project actually exists
    try:
        _ = api.project(project_path)
        project_exists = True
    except ValueError:
        project_exists = False

    # Define the parameter lists
    batch_sizes = args.batch_sizes
    epoch_lens = args.epochs
    ks = args.ks
    data_files = args.data_files
    model_names = args.model_names
    finetunes = args.finetunes
    finetune_scripts = args.finetune_scripts
    hidden_dropout = args.hidden_dropouts
    attn_dropout = args.attn_dropouts
    lrs = args.lrs
    w_decays = args.decays
    stoch_depths = args.stoch_depths
    label_smooth_vals = args.label_smoothing

    for i in range(0, args.max_loops):
        for  dataset, finetune, script, bs, epochs, k, h_dropout, a_dropout, lr, decay, stoch_depth, label_smooth_val, model_name in itertools.product(data_files, finetunes, finetune_scripts, batch_sizes, epoch_lens, ks, hidden_dropout, attn_dropout, lrs, w_decays, stoch_depths, label_smooth_vals, model_names):

            data_file = {"train": dataset}
            dataset_type = os.path.splitext(data_file["train"])[0]
            
            print(f"\nRunning fine-tune with finetune={finetune}, batch_size={bs}, data_file={dataset}, model_name={model_name}, lr={lr}, decay={decay}, hd={h_dropout}, ad={a_dropout}")

            cmd = [
                "python", script,
                "--batch_size", str(bs),
                "--epochs", str(epochs),
                "--k", str(k),
                "--data_file", dataset,
                "--model_name", model_name,
                "--finetune", finetune,
                "--hidden_dropout", str(h_dropout),
                "--attn_dropout", str(a_dropout),
                "--lr", str(lr),
                "--decay", str(decay),
                "--stoch_depth", str(stoch_depth),
                "--label_smoothing", str(label_smooth_val),
                "--project_title", project_title,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            
            if result.stderr:
                print("Error:", result.stderr)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model")
    parser.add_argument(
        "--wandb_proj",
        type=str,
        # Insert a string of your own Wandb project path here with "username/projectname"
        required=True,
        help="Current Wandb project path"
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[4],
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[60],
        help="Number of epochs for fine-tuning"
    )
    parser.add_argument( # K-fold cross validation is no longer utilized, but for simplicity has been set to 1
        "--ks",
        type=int,
        nargs="+",
        default=[1],
        help="Number of folds for K-fold cross validation"
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=["org_vid_path_100_train.csv"],
        help="Path to the train data CSV files"
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["facebook/timesformer-base-finetuned-k400"],
        help="Pretrained model or saved fine-tuned model name to use"
    )
    parser.add_argument( 
        "--finetunes",
        type=str,
        nargs="+",
        default=["default"],
        help="Fine-tuning method"
    )
    parser.add_argument(
        "--hidden_dropouts",
        type=float,
        nargs="+",
        choices=[0.0, 
            0.05,
            0.1, 
            0.15,
        ],
        default=[0.0],
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler"
    )
    parser.add_argument(
        "--attn_dropouts",
        type=float,
        nargs="+",
        choices=[0.0, 
            0.05,
            0.1, 
            0.15,
        ],
        default=[0.0],
        help="The dropout ratio for the attention probabilities"
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        choices=[0.0001,
           0.00005,
           0.00001
        ],
        default=[0.0001],
        help="The learning rate"
    )
    parser.add_argument(
        "--decays",
        type=float,
        nargs="+",
        choices=[0.1,
            0.01,
            0.05,
            0.001,
            0.0001
        ],
        default=[0.01],
        help="The weight decay"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        nargs="+",
        choices=[0.0,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3
        ],
        default=[0.3],
        help="Label smoothing for cross entropy loss"
    )
    parser.add_argument(
        "--stoch_depths",
        type=float,
        nargs="+",
        choices=[0.0,
            0.1,
            0.2,
            0.3
        ],
        default=[0.0],
        help="Stochastic depth, which will randomly drop entire transformer layers during training to encourage robustness"
    )
    parser.add_argument(
        "--finetune_scripts",
        type=str,
        nargs="+",
        default=["seedless-finetune.py"],
        help="Finetuning script, but can add more for other variations of the script"
    )
    parser.add_argument(
        "--max_loops",
        type=int,
        default=1,
        help="The number of times the for loop will loop; each loop means another iteration of / saved model with each combination of optimization parameters"
    )
    args = parser.parse_args()
    main(args)