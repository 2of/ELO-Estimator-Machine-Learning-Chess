# Chess ELO Estimator

This free tool provides an alternative to Chess.com's ELO estimation, allowing you to get an approximate ELO rating based on your PGN files.

## Try It Out

You can test the estimator with your own PGN file by visiting the web interface at:

* **[2of.io/#/chesselo](https://2of.io/#/chesseelo)**

## How It Works

The core of this estimator is a few machine learning models trained on a massive dataset of chess games.

* **Data Ingestion**: The system ingests standard **PGN (Portable Game Notation)** files.
* **Tokenization**: We process the move sequences, treating each move as a unique token. This tokenization is performed using a simple hashmap of all the distinct moves found within the training dataset.
* **Training**: The model was trained on a **33GB file** containing game data from **Lichess.com**.



## on Tokenization:

```bash
python3 src/vis/vismoves.py
```
Will create a nifty bunch of stats. A pre-computed summary of the move frequencies, acting as their hashmap is in */data/MovesHash* (csv & pandas)



## Local Usage

For those who want to run the estimator locally, you can use the provided Python script.

### Prerequisites

First, install the required Python libraries:

```bash
pip3 install -r requirements.txt