import pandas as pd

# Load the saved file
df = pd.read_pickle("./data/MovesHash/MovesHash.pd")
print(df.head())        # first 5 rows
print(df.info())        # check types and non-null counts

# Optional: verify index is 'row'
print(df.index.name)    # should print 'row'