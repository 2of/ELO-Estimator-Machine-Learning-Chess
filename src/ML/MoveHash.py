import pandas as pd

class MoveHashHandler:
    def __init__(self, hashfile):
        self.hashfile = hashfile
        self.df = pd.DataFrame()
        self.hashmap = {}

    def load_map(self):
        """Load the pandas pickle file and populate the hashmap {move: row}"""
        try:
            self.df = pd.read_pickle(self.hashfile)
        except Exception as e:
            print(f"Failed to open the pandas file for the hashmap: {e}")
            self.df = pd.DataFrame()
            self.hashmap = {}
            return

        # Build hashmap using the index as row
        if not self.df.empty:
            self.hashmap = dict(zip(self.df['move'], self.df.index))
        else:
            self.hashmap = {}

    def show_df(self):
        """chonky"""
        print(self.df)

    def show_hashmap(self):
        """Print the move -> row hashmap"""
        print(self.hashmap)

    def get(self, move):

        return self.hashmap.get(move, None)


if __name__ == "__main__":
    handler = MoveHashHandler("./data/MovesHash/Moveshash.pd")
    handler.load_map()
    handler.show_df()
    # handler.show_hashmap()

    # Example usage
    print(handler.get("g6"))
    print(handler.get("Qg1f1"))