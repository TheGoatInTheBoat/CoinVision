# CoinVision
CoinVision is an open-source computer vision and machine learning project focused on automatically analyzing photographs of coins. The goal of the project is to develop an AI system capable of identifying coins on a wide scale and at a level of detail matching that of a numismatist.
Dataset Setup

This project uses the US Coins dataset from Kaggle.

1. Install the Kaggle CLI
pip install kaggle


Make sure your Kaggle API credentials are configured before running the script.

2. Download the dataset

Run the following command from the project root:

python download_dataset.py


The script will automatically:

Create the datasets/ folder if it doesn't exist.

Download the US Coins dataset from Kaggle.

Unzip the dataset into the datasets/ folder.

After downloading, your project should look something like this:

project/
├── datasets/
│   └── ...
├── download_dataset.py
├── .gitignore
└── README.md


The datasets/ folder is included in .gitignore, so the downloaded dataset will not be committed to Git.