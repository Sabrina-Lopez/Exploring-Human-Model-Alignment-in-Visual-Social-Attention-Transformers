# Exploring-Human-Model-Alignment-in-Visual-Social-Attention-Transformers
Transformers Work for "Exploring Human-Model Alignment in Visual Social Attention During Help-and-Hinder Social Interaction Classification"

## How to get started
To start, you will download the video dataset, setup a Wandb account, and setup an environment with requirements.

### Video dataset
For the first time, head to the [workshop paper's page](https://humanmodelvsa.github.io/HumanModelVSA) and downloading the Social Interactions Gaze Dataset. In the directory of your copy of this repository, create a folder called "videos". For the downloaded zip file, extract the files, go into the extracted files, traverse to the videos (i.e., "Stimuli dataset/Stimuli dataset/Videos"), and copy them to your "videos" folder.

### Wandb account
If you do not have a Wandb account already, you will then set up an account to see the training, validation, and testing accuracies and losses alongside other data in the fine-tuning and testing. After creating an account, get your API key by clicking your profile picture and clicking "API key", and get your username or, if you decide to make a team for collaboration, your team entity name. You can then create a .env file like the following:

```bash
WANDB_USERNAME = ["YOUR USERNAME / TEAM ENTITY NAME GOES HERE"]
WANDB_API_KEY = ["YOUR API KEY GOES HERE"]
```

### Environment
To setup your virtual or conda environment, traverse to the directory containing the requirements.txt file, open your terminal, and paste the following command in the terminal:

```bash
pip install -r requirements.txt
```

## Fine-tuning
Once everything is setup, you can run `text-to-data.py` to get the videos of the dataset organized into the training/testing split that we utilized. After, run `create-csv.py` to have .csv files of the training set and the testing set. You can also utilize the `data_augment.py` script to create augmented data that will double the dataset, but, if you are looking to replicate the results of the paper, just the original dataset is needed.

```bash
python text-to-data.py
python create-csv.py
```

You can either run `automate-finetunes.py` or `seedless-finetune.py` to run the fine-tuning of the TimeSformer model. The major benefit of the former is that you can add to the different parameters, add custom script variations, more Hugging Face models, etc. and run every combination as needed. Plus, you can run each combination a number of times with the "max_loops" argument, so you can, for instance, run each combination 30 times to get more data to compute a mean of the training and validation accuracy later. While the fine-tuning is occuring, you can watch the progress of the fine-tuning in real-time from your Wandb project.

```bash
python automate-finetunes.py --wandb_proj "yourUsername/yourProjectName"
python seedless-finetune.py --project_title "yourProjectName"
```

## Testing
You can either run `automate-tests.py` or `seedless-test.py` to run the testing of the TimeSformer model. The latter has the same purpose as automate-finetunes.py, but it can also grab all your saved models and test them rather than individually testing each one via the seedless-test.py script. You can using the testing to tests non-fine-tuned Hugging Face models or, with a file path, fine-tuned Hugging Face models.

```bash
python automate-tests.py --wandb_proj "yourUsername/yourProjectName"
python seedless-test.py --project_title "yourProjectName"
```

## Visualize Attention
To visualize the attention of a model, get a file path to a saved model and insert the path when running `visualize-model-attn.py`. The .npy files and video files that visually present the heatmaps will be available in a folder called "attn_outputs" unless modified otherwise.

```bash
python visualize-model-attn.py --model_name "path/to/your/saved/model"
```