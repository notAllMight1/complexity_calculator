import pandas as pd

def load_and_clean_data(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Fill or drop rows with missing values (example: drop here)
    df.dropna(inplace=True)
    # Optional: Normalize code strings (removing inconsistent spacing, etc.)
    df['code'] = df['code'].apply(lambda code: "\n".join(line.strip() for line in code.splitlines()))
    return df

if __name__ == "__main__":
    data_path = "./Dataset/Normal3lang.csv"  # adjust relative path as needed
    df = load_and_clean_data(data_path)
    print(df.head())
