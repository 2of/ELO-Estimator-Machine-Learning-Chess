class LoadGamesFromDirectoryWrapper(): 
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.fileIndex = 0
        self.fileListing = self.get_all_files_in_dir()
        self.numFiles = lambda: len(self.fileListing) 
        self.openfile = self.load_file(self.fileListing[0]) if self.fileListing else None

    def get_all_files_in_dir(self):
        import os
        files = []
        for entry in os.scandir(self.file_dir):
            if entry.is_file() and entry.name.endswith(".txt"):
                files.append(entry.path)
        return sorted(files)

    def load_file(self, filename):
        """
        Load a file, strip newlines, ignore empty lines,
        and split each line by commas into a list of values.
        """
        with open(filename, "r", encoding="utf-8") as f:
            lines = [
                line.strip().split(",")  # split CSV-style
                for line in f
                if line.strip()  # skip empty/whitespace-only lines
            ]
        return lines
    def get_next_file(self):
        self.fileIndex += 1
        if self.fileIndex >= self.numFiles():
            return None
        return self.fileListing[self.fileIndex]
        
    def print_summary(self):
        return f"There are {self.numFiles()} files in {self.file_dir}"



    # -----------------------
    # Iterables
    # -----------------------
    def __iter__(self):
        self.fileIndex = 0  # reset iteration
        return self

    def __next__(self):
        if self.fileIndex >= self.numFiles():
            raise StopIteration
        lines = self.load_file(self.fileListing[self.fileIndex])
        self.fileIndex += 1
        return lines


if __name__ == "__main__":
    Loader = LoadGamesFromDirectoryWrapper("../data/samples")   
    print(Loader.print_summary())

    # Iterate through all files
    for i, file_lines in enumerate(Loader):
        print(f"File {i+1}: {len(file_lines)} lines")
   