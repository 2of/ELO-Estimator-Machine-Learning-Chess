# Chess ELO Estimator

This free tool provides an alternative to Chess.com's ELO estimation, allowing you to get an approximate ELO rating based on your PGN files.

## Try It Out

You can test the estimator with your own PGN file by visiting the web interface at:

* **[ML Chess Elo Estimator](https://thingies.dev/#/ChessEloEsimator)** (currently not up)


## How It Works

The core of this estimator is a few machine learning models trained on a massive dataset of chess games.

* **Data Ingestion**: The system ingests standard **PGN (Portable Game Notation)** files.
* **Tokenization**: We process the move sequences, treating each move as a unique token. This tokenization is performed using a simple hashmap of all the distinct moves found within the training dataset.
* **Training**: The model was trained on a **33GB file** containing game data from **Lichess.com**, decompressed it's around 280gb Totalling 9 million or so games



Essentially > Chess games > Strip > Convert all to labels,data tensors (black,white elo), then sequence of moves.


For the LSTM and other attention based models we use tokenization. 

For the CNN model we just used computed gamestates (on gpu at train and at forward pass)



## on Tokenization:

```bash
python3 src/vis/vismoves.py
```
Will create a nifty bunch of stats. A pre-computed summary of the move frequencies, acting as their hashmap is in */data/MovesHash* (csv & pandas)


## models


Tensorflow models are in /models. They take in tensors of the form gxnummoves for lstm / attention models and gxnummovesx8x8 for single (g is num games)


## training

I just ran it through a few passes on my single 3080, so if you wanna fiddle with some hyperparameters etc be my guess!


## test and results

you can see a summary of results at

* **[2of.io/#/proj/elo](https://2of.io/#/proj/chessEloEstimator)** 


## Local Usage

For those who want to run the estimator locally, you can use the provided Python script.

```bash
python3 predict.py -pgnfile "path_to_your_pgn_file.pgn" -model "path_to_chosenmodel_model" 
```

Leaving either parameter blank will default to a dummy pgn / LSTM model

### Prerequisites

First, install the required Python libraries:

```bash
pip3 install -r requirements.txt
