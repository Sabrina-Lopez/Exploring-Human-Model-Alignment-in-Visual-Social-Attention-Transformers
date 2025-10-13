import subprocess
import itertools
import wandb
from dotenv import load_dotenv
import os
import random
import argparse


def main(args):
    api = wandb.Api()
    project_path = args.wandb_proj
    _, project_title = project_path.split('/', 1)
    saved_model_dir = args.saved_model_dir

    project_exists = False
    # Figure out once whether the project actually exists
    try:
        _ = api.project(project_path)
        project_exists = True
    except ValueError:
        project_exists = False

    # Define the parameter lists
    batch_sizes = args.batch_sizes
    data_files = args.data_files
    test_scripts = args.test_scripts

    # Recursively gather all model subfolders inside each folder in saved_model_dir
    model_names = []
    for root, dirs, _ in os.walk(saved_model_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            if os.path.isdir(full_path):
                model_names.append(os.path.relpath(full_path, saved_model_dir))

    for script, model_name, dataset, bs in itertools.product(test_scripts, model_names, data_files, batch_sizes):  

        data_file = {"test": dataset}
        dataset_type = os.path.splitext(data_file["test"])[0]

        # To ensure each test is unique, even if running through this script again with new tests but the same directory
        # """
        if project_exists:
            filters = {
                # "config.batch_size": bs,
                "config.model_name": str(os.path.join(saved_model_dir, model_name))
            }
            runs = api.runs(project_path, filters=filters)
            
            # for run in runs:
            #     config = run.config
            #     print(config, len(runs))
            
            if (len(runs) != 0): 
                # print('continue')
                continue
        # """

        print(f"\nRunning test with batch_size={bs}, data_file={dataset}, model_name={model_name}")
        cmd = [
            "python", script,
            "--batch_size", str(bs),
            "--data_file", dataset,
            "--model_name", str(os.path.join(saved_model_dir, model_name)),
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
        "--data_files",
        type=str,
        nargs="+",
        default=["org_vid_path_100_test.csv"],
        help="Path to the test data CSV files"
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default="./saved_model_dir",
        help="Directory of saved fine-tuned models"
    )
    parser.add_argument(
        "--test_scripts",
        type=str,
        nargs="+",
        default=["seedless-test.py"],
        help="Testing script, cbut can add more for other variations of the script"
    )
    args = parser.parse_args()
    main(args)