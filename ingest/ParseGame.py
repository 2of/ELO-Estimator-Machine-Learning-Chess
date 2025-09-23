from .LoadGamesFromFilesWrapper import *
'''
Parsees games to create a hash of all possible moves
'''
class MultiFileLoaderWrapper:
    """
    Wrapper to load games across multiple files.
    Supports fetching n games at a time, potentially spanning multiple files.
    """
    def __init__(self, file_dir):
        self.loader = LoadGamesFromDirectoryWrapper(file_dir)
        self.file_index = 0
        self.current_file_lines = []
        self.current_line_index = 0
    def print_summary(self):
        return self.loader.print_summary()
    def _load_next_file(self):
        """Load the next file into memory"""
        if self.file_index >= self.loader.numFiles():
            return False
        self.current_file_lines = self.loader.load_file(self.loader.fileListing[self.file_index])
        self.current_line_index = 0
        self.file_index += 1
        return True

    def get_n_games(self, n):
        """Return up to n games across files"""
        games = []

        while len(games) < n:
            # If current file exhausted, load next
            if self.current_line_index >= len(self.current_file_lines):
                if not self._load_next_file():
                    break  # no more files
            remaining = len(self.current_file_lines) - self.current_line_index
            to_take = min(n - len(games), remaining)
            games.extend(self.current_file_lines[self.current_line_index:self.current_line_index + to_take])
            self.current_line_index += to_take

        return games

    def reset(self):
        """Reset loader to start from first file"""
        self.file_index = 0
        self.current_file_lines = []
        self.current_line_index = 0


if __name__ == "__main__":
    wrapper = MultiFileLoaderWrapper("../data/samples")

    batch_size = 50
    total_games = 0
    
    while False:
        games = wrapper.get_n_games(batch_size)

        if not games:
            break
        total_games += len(games)
        print(f"Fetched {len(games)} games, total so far: {total_games}")

    print(f"All done! Total games loaded: {total_games}")