import json

outpath = "../data/moves_map.json"

from plaintextloaderhelper import PlaintextLoaderHelper

class MovesHash: 
    def __init__(self):
        self.map = {}
    def add_move(self, move):
        move = move.replace("+","").replace("#","").replace("\n","")
        if move not in self.map:
            self.map[move] = len(self.map) + 1
        return self.map[move]
    
    def print_map(self):
        for move, idx in self.map.items():
            print(f"{idx}: {move}")

mh = MovesHash()
Ploader = PlaintextLoaderHelper("../data/txt")

for file in Ploader:
    lines = Ploader.loadfile(file)
    for line in lines:
        for move in line.split(" ")[2:]:
            mh.add_move(move)

mh.print_map()

with open(outpath, "w", encoding="utf-8") as f:
    json.dump(mh.map, f, ensure_ascii=False, indent=2)