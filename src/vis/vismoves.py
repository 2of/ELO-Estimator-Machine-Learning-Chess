import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import seaborn as sns


sns.set_theme(style="whitegrid", palette="dark")

here = Path(__file__).resolve().parent  


repo_root = here.parent.parent  


data_path = repo_root / "data" / "MovesHash" / "MovesHash.csv"
df = pd.read_csv(data_path)

print(f"✅ Loaded {len(df)} rows from {data_path}")
# Sort by count
df_sorted = df.sort_values(by="count", ascending=False)

# === Top 10 most common moves ===
top10 = df_sorted.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x="move", y="count", data=top10, palette="viridis")

# Annotate each bar
for i, val in enumerate(top10["count"]):
    plt.text(i, val, f"{val:,}", ha="center", va="bottom", fontsize=10, weight="bold")

plt.title("🔥 Top 10 Most Common Chess Moves 🔥", fontsize=16, weight="bold")
plt.xlabel("Move")
plt.ylabel("Count")
plt.show()

# === Bottom 10 least common moves ===
bottom10 = df_sorted.tail(10)
plt.figure(figsize=(10, 6))
sns.barplot(x="move", y="count", data=bottom10, palette="magma")

for i, val in enumerate(bottom10["count"]):
    plt.text(i, val, f"{val:,}", ha="center", va="bottom", fontsize=10, weight="bold")

plt.title("💤 10 Least Common Chess Moves 💤", fontsize=16, weight="bold")
plt.xlabel("Move")
plt.ylabel("Count")
plt.show()

# === Histogram of counts (log scale) ===
plt.figure(figsize=(10, 6))
sns.histplot(df["count"], bins=50, log_scale=True, color="steelblue")
plt.title("📊 Distribution of Move Frequencies", fontsize=16, weight="bold")
plt.xlabel("Move Count")
plt.ylabel("Frequency (log scale)")
plt.show()

# === Rank vs Count Scatter (Zipf-like curve) ===
df_sorted["rank"] = range(1, len(df_sorted) + 1)

plt.figure(figsize=(10, 6))
plt.loglog(df_sorted["rank"], df_sorted["count"], marker="o", linestyle="", alpha=0.5, color="darkorange")
plt.title("⚡ Rank vs Count (Zipf’s Law in Chess Moves) ⚡", fontsize=16, weight="bold")
plt.xlabel("Rank of Move (1 = most common)")
plt.ylabel("Count (log scale)")
plt.show()

# === Word Cloud of moves ===
word_freq = dict(zip(df["move"], df["count"]))
wordcloud = WordCloud(width=1000, height=600, 
                      background_color="black",
                      colormap="Spectral", 
                      max_words=200).generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("🌍 Word Cloud of Chess Moves 🌍", fontsize=16, weight="bold", color="white")
plt.show()

# === Print summary ===
print("Most common move:", df_sorted.iloc[0].to_dict())
print("Least common move:", df_sorted.iloc[-1].to_dict())
print(f"Dataset has {len(df)} unique moves, total of {df['count'].sum():,} moves recorded!")