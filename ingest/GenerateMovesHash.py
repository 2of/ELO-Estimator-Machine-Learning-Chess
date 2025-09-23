import csv
import json
from ParseGame import MultiFileLoaderWrapper

outpath = "../data/moves_map2.csv"

class MovesHash:
    def __init__(self):
        self.map = {}       # move -> index
        self.counts = {}    # move -> count

    def add_move(self, move):
        move = move.replace("+", "").replace("#", "").replace("\n", "")
        if move not in self.map:
            self.map[move] = len(self.map) + 1
            self.counts[move] = 1
        else:
            self.counts[move] += 1
        return self.map[move]

    def print_map(self):
        for move, idx in self.map.items():
            print(f"{idx}: {move} ({self.counts[move]} times)")

    def write_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row", "move", "count"])
            for move, idx in self.map.items():
                writer.writerow([idx, move, self.counts[move]])


# --------------------------
# Main processing
# --------------------------
mh = MovesHash()
wrapper = MultiFileLoaderWrapper("/volumes/bck/LICHESS")

c = 0

print(wrapper.print_summary())
while True:
    print("We are on chunk" , c)
    # break
    c += 1
    games = wrapper.get_n_games(1000)
    if not games:
        break
    for line in games:
        # skip first two tokens if needed, then add moves
        for move in line.split(" ")[2:]:
            mh.add_move(move)

# print summary
mh.print_map()

# write to CSV
mh.write_csv(outpath)
print(f"Saved moves CSV to {outpath}")