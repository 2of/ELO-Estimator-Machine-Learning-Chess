import json
from plaintextloaderhelper import *


class PGNStreamer:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.file = None

    def load_file(self, file_path=None):
        """Open a PGN file for streaming"""
        if file_path:
            self.file_path = file_path
        if not self.file_path:
            raise ValueError("No file path provided to load.")
        self.file = open(self.file_path, "r", encoding="utf-8", errors="ignore")

    def get_number_lines_of_file(self):
        """Return total number of lines in the current file"""
        if not self.file_path:
            raise RuntimeError("File not loaded.")
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)

    def reset_file(self):
        """Reset file pointer back to start"""
        if not self.file:
            raise RuntimeError("File not loaded. Call load_file() first.")
        self.file.seek(0)

    def get_all_games(self):
        """Read all games from the file into a list"""
        games = []
        while True:
            game = self.get_next_game()
            if game is None:
                break
            games.append(game)
        return games

    def get_next_n_games(self, n):
        """Return up to n games from the current file as a list"""
        games = []
        for _ in range(n):
            game = self.get_next_game()
            if not game:
                break
            games.append(game)
        return games

    def get_next_game(self):
        """Return the next game as a raw string"""
        if not self.file:
            raise RuntimeError("File not loaded. Call load_file() first.")

        game_lines = []
        while True:
            line = self.file.readline()
            if not line:
                break

            if line.strip() == "" and game_lines:
                pos = self.file.tell()  # save current file pointer
                next_line = self.file.readline()

                if not next_line:
                    break

                if next_line.startswith('['):  # start of new game
                    self.file.seek(pos)  # rewind so next call starts here
                    return "".join(game_lines)
                else:
                    game_lines.append(line)
                    game_lines.append(next_line)
            else:
                game_lines.append(line)

        if game_lines:
            return "".join(game_lines)

        # End of file reached
        self.file.close()
        self.file = None
        return None

    def __iter__(self):
        """Allow: for game in streamer:"""
        self.reset_file()
        return self

    def __next__(self):
        game = self.get_next_game()
        if not game:
            raise StopIteration
        return game

    def __del__(self):
        if self.file:
            self.file.close()


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def load_Moves_to_map(): 
    pass



class FilesasLinesWrapper():
    def __init__(self, files = None):
        self.files = files  
        self.currentFileIndex = 0
        self.current_streamer = None


    def get_next_game(self):
        """Fetch next game across multiple files"""
        while True:
            if not self.current_streamer:
                if not self._open_next_file():
                    return None  # no more files

            game = self.current_streamer.get_next_game()
            print("load in")
            if game:
                return game
            else:
                # current file exhausted, move on
                self._open_next_file()


    def _open_next_file(self):
        if self.current_streamer:
            del self.current_streamer
            self.current_streamer = None

        if self.currentFileIndex >= len(self.files):
            return False

        self.current_streamer = PGNStreamer(self.files[self.currentFileIndex])
        self.current_streamer.load_file()
        self.currentFileIndex += 1
        return True
    
    def get_next_n_games(self, n):
        """Fetch up to n games across files"""
        games = []
        for _ in range(n):
            game = self.get_next_game()
            if not game:
                break
            games.append(game)
        return games

    def reset(self):
        """Reset to start of all files"""
        if self.current_streamer:
            del self.current_streamer
        self.currentFileIndex = 0
        self.current_streamer = None

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        game = self.get_next_game()
        if not game:
            raise StopIteration
        return game

if __name__ == "__main__":
    # Load all file paths
    allfiles = PlaintextLoaderHelper("../data/samples")
    wrapper = FilesasLinesWrapper(allfiles)

    # Count all games
    total_games = 0
    for game in wrapper:
        total_games += 1

    print(f"Total games across all files: {total_games}")