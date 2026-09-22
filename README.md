# CoinVision
CoinVision is an open-source computer vision and machine learning project focused on automatically analyzing photographs of coins. The goal of the project is to develop an AI system capable of identifying coins on a wide scale and at a level of detail matching that of a numismatist.
Dataset Setup

This project uses the US Coins dataset from Kaggle.

1. Install the Kaggle CLI
pip install kaggle


Make sure your Kaggle API credentials are configured before running the script.

2. Create a Kaggle API Token

Sign in to your Kaggle account and create an API token from your Kaggle account settings.

Kaggle provides an access token in the following format:

###KGAT_...

3. Save Your Kaggle Token

Create the Kaggle configuration directory:

New-Item -ItemType Directory -Force "$HOME\.kaggle"

Create the token file:

notepad "$HOME\.kaggle\access_token"

Paste your Kaggle token into the file and save it.

4. Download the Dataset

From the root directory of this project, run:

python -m kaggle datasets download -d sergiosaharovskiy/uscoins -p datasets --unzip

This will download the dataset to CoinVision/datasets/

The datasets/ folder is included in .gitignore, so the downloaded dataset will not be committed to Git.