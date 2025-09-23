class PlaintextLoaderHelper:
    ''' handles loading many files of plaintext games '''
    def __init__(self, file_path):
        self.file_path = file_path
        self.current_file_index = 0
        self.files = self.get_files_in_dir()


    def get_files_in_dir(self):
        import os
        files = []
        for entry in os.scandir(self.file_path):
            if entry.is_file() and entry.name.endswith(".txt"):
                files.append(entry.path)
        print(files)
        return files

    

    def loadfile(self,filename):
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines
    

    def get_next_file(self):
        if self.current_file_index >= len(self.files):
            return None
        file = self.files[self.current_file_index]
        self.current_file_index += 1
        return file

    def __iter__(self):
        self.current_file_index = 0  # reset for new 
        return self

    def __next__(self):
        '''
        im learninding 
        
        '''
        if self.current_file_index >= len(self.files):
            raise StopIteration
        file = self.files[self.current_file_index]
        self.current_file_index += 1
        return file


Ploader = PlaintextLoaderHelper("../data/samples")

for file in Ploader:
    lines = Ploader.loadfile(file)
    print(file, len(lines))