import json
import re
from filestream import PGNStreamer
import os
'''
Strips data of metadata we don't care about, as 33m games is ~212gb uncompressed otherwise
a bit of a waste of time considering we could simply preprocess in ... .



'''


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

import re

def extract_elo_and_moves(game_text):
    white_elo = None
    black_elo = None

    white_match = re.search(r'\[WhiteElo\s+"(\d+)"\]', game_text)
    if white_match:
        white_elo = white_match.group(1)

    black_match = re.search(r'\[BlackElo\s+"(\d+)"\]', game_text)
    if black_match:
        black_elo = black_match.group(1)

    parts = game_text.split("\n\n", 1)
    moves_text = parts[1].strip() if len(parts) > 1 else ""

    #  clock/time annotations and any comments in  {}
    moves_text = re.sub(r'\{[^}]*\}', '', moves_text)
    # Remove result  ie 1-0, 0-1, 1/2-1/
    moves_text = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', moves_text)
    # Remove move numbers (like 1. or 1... or 1.)
    moves_text = re.sub(r'\d+\.(\.\.)?', '', moves_text)
    #blunder notaiton and so on
    moves_text = re.sub(r'(\.\.\.|[?!]+)', '', moves_text)
    # Split moves by whitespace, filter out empty strings
    moves_list = [m for m in moves_text.split() if m.strip()]

    return white_elo, black_elo, moves_list

def main():
    cfg = load_config()
    streamer = PGNStreamer(file_path=cfg["file_path"])
    streamer.load_file()

    k = 100000  # Number of games per output file
    batch = []
    file_index = 1
    count = 0
    output_dir = cfg.get("output_dir_converted_filepath", ".")  
    for _ in range(33_000_000):  # large upper limit, breaks on None
        game = streamer.get_next_game()
        if game is None:
            break
        count += 1

        white_elo, black_elo, moves = extract_elo_and_moves(game)

        # Format the reduced game as a text block
        game_text = f"{white_elo}, {black_elo},{' '.join(moves)}\n"

        batch.append(game_text)

        if count % k == 0:
            filename = os.path.join(output_dir, f"reduced_games_{file_index}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(batch))
            print(f"Wrote {k} games to {filename}")
            batch = []
            file_index += 1

    # Write leftover games if any
    if batch:
        filename = f"reduced_games_{file_index}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(batch))
        print(f"Wrote {len(batch)} games to {filename}")

    print(f"Total games processed: {count}")


if __name__ == "__main__":
    main()
