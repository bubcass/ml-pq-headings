import pandas as pd

# ---- CONFIG ----
INPUT_FILE = "outputs/predictions_distilbert_latest.csv"  # change if needed
HEADING_COLUMN = "heading"  # change if needed

# ---- LOAD DATA ----
df = pd.read_csv(INPUT_FILE)

# ---- COUNT HEADINGS ----
counts = (
    df[HEADING_COLUMN].value_counts().rename_axis("heading").reset_index(name="count")
)

# ---- CALCULATE SHARES ----
total = counts["count"].sum()

counts["share"] = counts["count"] / total
counts["cum_share"] = counts["share"].cumsum()
counts["rank"] = range(1, len(counts) + 1)


# ---- FUNCTION TO FIND COVERAGE ----
def headings_for_threshold(threshold):
    row = counts[counts["cum_share"] >= threshold].iloc[0]
    return int(row["rank"])


# ---- PRINT RESULTS ----
thresholds = [0.50, 0.80, 0.90, 0.95]

print("\nPQ HEADING CONCENTRATION\n")

for t in thresholds:
    n = headings_for_threshold(t)
    print(f"{int(t * 100)}% of PQs are covered by the top {n} headings")

print(f"\nTotal headings in taxonomy: {len(counts)}")
print(f"Total PQs analysed: {total}")

# ---- SAVE FULL TABLE ----
counts.to_csv("heading_distribution.csv", index=False)

print("\nSaved full distribution to: heading_distribution.csv")
