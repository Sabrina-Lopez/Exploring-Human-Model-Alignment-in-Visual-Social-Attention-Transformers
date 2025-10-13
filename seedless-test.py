import numpy as np
import random
import os
import torch
import wandb
import argparse
from dotenv import load_dotenv
from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from utils import preprocess_function, custom_collate_fn, evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import re


load_dotenv()
wandb.login(key = os.environ.get("WANDB_API_KEY"))


def main(args):
    seedlessBool = True

    """ Snippet of code to establish a seed if needed
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seedlessBool = False
    """


    # Retrieve and enable the loading of the test data
    data_files = {"test": args.data_file}
    data_csv = load_dataset('csv', data_files=data_files, sep=',')

    data_csv["test"] = data_csv["test"].map(preprocess_function)
    test_dataset = data_csv["test"].with_format(
        "torch", columns=["video_path", "start", "end", "label"]
    )

    batch_size = args.batch_size
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )


    # Define custom labels
    custom_labels = ["help", "hinder", "physical"]
    num_custom_labels = len(custom_labels)

    # Create label2id and id2label dicts
    label2id = {"help": 0, "hinder": 1, "physical": 2}
    id2label = {0: "help", 1: "hinder", 2: "physical"}

    # Load the base config from the pretrained checkpoint
    base_model_name = args.model_name

    # Load the base config from the pretrained checkpoint
    config = TimesformerConfig.from_pretrained(base_model_name)

    # Update number of labels and label mappings
    config.num_labels = num_custom_labels
    config.id2label = id2label
    config.label2id = label2id
    config.num_frames = 16


    # Get the pretrained image processor and model
    image_processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = TimesformerForVideoClassification.from_pretrained(base_model_name, config=config, ignore_mismatched_sizes=True)


    dataset_type = os.path.splitext(args.data_file)[0]

    name = None
    prefix = 'test_data_'
    substring = dataset_type[len(prefix):]
    name = "times_" + substring

    # Default as in the default fine-tuning
    if ("default" in args.model_name):
        finetune_method = "default"
        finetune_bool = True
    else:
        finetune_method = "none"
        finetune_bool = False


    pattern = re.compile(
        r"-bs-(?P<bs>\d+)"
    )
    m = pattern.search(args.model_name)
    bs = None
    if not m:
        print(f"Couldn't parse hyperparams from '{args.model_name}'. Likely because it is not fine-tuned.")
    else: 
        bs_s = m.group("bs")   # e.g. "16"
        bs = int(bs_s)


    # Initialize wandb with project and configuration details
    wandb.init(
        project=args.project_title,
        config={
            "seedlessBool": seedlessBool,
            "train_batch_size": bs,
            "test_batch_size": batch_size,
            "data_type": dataset_type,
            "model_name": base_model_name,
            "finetune_method": finetune_method,
            "finetune_bool": finetune_bool
        },
        name=name
    )


    # Evaluate the testing dataset
    all_preds, all_labels, losses, accuracies = evaluate(model, test_loader, image_processor)

    accuracy = accuracy_score(all_labels, all_preds)
    # Get classification report as a dictionary
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=["help", "hinder", "physical"],
        output_dict=True
    )

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["help", "hinder", "physical"]))


    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        # "test_batch_size": batch_size,
        # Per-class metrics
        "precision_help": report_dict["help"]["precision"],
        "precision_hinder": report_dict["hinder"]["precision"],
        "precision_physical": report_dict["physical"]["precision"],
        "recall_help": report_dict["help"]["recall"],
        "recall_hinder": report_dict["hinder"]["recall"],
        "recall_physical": report_dict["physical"]["recall"],
        "f1_help": report_dict["help"]["f1-score"],
        "f1_hinder": report_dict["hinder"]["f1-score"],
        "f1_physical": report_dict["physical"]["f1-score"],
        "support_help": report_dict["help"]["support"],
        "support_hinder": report_dict["hinder"]["support"],
        "support_physical": report_dict["physical"]["support"],
        # Macro averages
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        # Weighted averages
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"]
    })


    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on a test dataset")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="org_vid_path_100_test.csv",
        help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # Insert the file path of a saved model here, fine-tuned or not fine-tuned
        required=True,
        help="Pretrained model or saved fine-tuned model name to use"
    )
    parser.add_argument(
        "--project_title",
        type=str,
        # Insert a string of your own Wandb project name here like "projectname"
        default="October Fine-tuned Intent Classification Testing",
        help="Title for the current Wandb project"
    )
    args = parser.parse_args()
    main(args)