import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Define the paths to the JSON files
json_files = ["dclm_bert_natural.json", "dclm_ft_natural.json"]

# Load and combine data from all JSON files
all_data = []
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract filename without extension for mix label
    mix_name = Path(json_file).stem

    # Add mix information to each domain entry
    for item in data:
        item["mix"] = mix_name
        all_data.append(item)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Set up the plot style
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")

# Create histogram with color coding by mix and better spacing
ax = sns.barplot(data=df, x="domain", y="weight", hue="mix", palette="Set2")

# Add spacing between domain groups
ax.tick_params(axis="x", which="major", pad=10)

# Customize the plot
plt.title("Domain Weight Distribution Across DCLM Mixes", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Domain", fontsize=12, labelpad=15)
plt.ylabel("Weight", fontsize=12, labelpad=15)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)

# Add vertical lines to separate domain groups every 3 domains for better readability
for i in range(0, len(df["domain"].unique()), 1):
    plt.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

# Improve legend positioning and styling
plt.legend(title="Mix", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fancybox=True, shadow=True)

# Adjust layout to prevent label cutoff with more padding
plt.tight_layout(pad=2.0)

# Show the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics by Mix:")
print(df.groupby("mix")["weight"].agg(["count", "mean", "std", "min", "max"]))

# Print top domains by weight for each mix
print("\nTop 5 domains by weight for each mix:")
for mix in df["mix"].unique():
    mix_data = df[df["mix"] == mix].nlargest(5, "weight")
    print(f"\n{mix}:")
    for _, row in mix_data.iterrows():
        print(f"  {row['domain']}: {row['weight']:.4f}")
