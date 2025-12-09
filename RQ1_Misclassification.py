import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("./Results/glare_predictions.csv")

# Filter only misclassified samples
misclassified = df[df["ground_truth"] != df["prediction"]]

# Count how many times each true label was misclassified
error_counts = misclassified["ground_truth"].value_counts().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
error_counts.plot(kind='bar', color='blue', edgecolor='black')
plt.xlabel("Ground Truth Label")
plt.ylabel("Number of Misclassifications")
plt.grid(alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("./Results/figures/test1.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(misclassified["confidence"], bins=20, color='orange', edgecolor='black')
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.grid(alpha=0.5)
plt.savefig("./Results/figures/test2.png")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=misclassified, x="ground_truth", y="confidence")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(alpha=0.5)
plt.savefig("./Results/figures/test3.png")
plt.show()

# Compute total misclassifications per ground truth class
misclassified = df[df["ground_truth"] != df["prediction"]]
top_classes = misclassified["ground_truth"].value_counts().nlargest(5).index.tolist()

# Filter only rows where ground truth or prediction is in top 5
subset_df = df[(df["ground_truth"].isin(top_classes)) | (df["prediction"].isin(top_classes))]

# Compute confusion matrix
cm = confusion_matrix(
    subset_df["ground_truth"],
    subset_df["prediction"],
    labels=top_classes
)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_classes)
disp.plot(ax=ax, cmap="Reds", colorbar=True)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("./Results/figures/test4.png")
plt.show()
