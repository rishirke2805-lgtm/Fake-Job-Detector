import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "dataset/cleaned_fake_job_postings.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Text columns we will use
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Replace missing values with empty text
for column in text_columns:
    df[column] = df[column].fillna("")

# Combine all important text fields
df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Apply cleaning
df["combined_text"] = df["combined_text"].apply(clean_text)

# Remove completely empty text records
df = df[df["combined_text"].str.len() > 0]

# Keep only required columns
processed_df = df[
    [
        "job_id",
        "combined_text",
        "fraudulent"
    ]
]

# Save processed dataset
processed_df.to_csv(
    "dataset/processed_jobs.csv",
    index=False
)

print("Processed dataset shape:", processed_df.shape)

# X = input text
X = processed_df["combined_text"]

# y = target
y = processed_df["fraudulent"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nPhase 2 preprocessing completed!")