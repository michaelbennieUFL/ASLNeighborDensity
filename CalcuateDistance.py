import pandas as pd


def load_sign_dataset(filepath):
    # Load only the required columns if they exist in the file
    cols = [
        "LemmaID", "Code", "SelectedFingers.2.0", "Flexion.2.0", "FlexionChange.2.0",
        "Spread.2.0", "SpreadChange.2.0", "ThumbPosition.2.0", "ThumbContact.2.0",
        "SignType.2.0", "Movement.2.0", "RepeatedMovement.2.0", "MajorLocation.2.0",
        "MinorLocation.2.0", "SecondMinorLocation.2.0", "Contact.2.0",
        "NonDominantHandshape.2.0", "UlnarRotation.2.0"
    ]
    # Determine available columns in the file
    available_cols = pd.read_csv(filepath, nrows=0, encoding="latin1").columns
    used_cols = [c for c in cols if c in available_cols]
    df = pd.read_csv(filepath, usecols=used_cols, encoding="latin1")
    return df


def sign_similarity(row1, row2, feature_cols):
    """
    Returns the number of matches and total compared features (ignoring both-NA pairs)
    and a list of feature-wise comparison (feature, row1_value, row2_value, match).
    """
    results = []
    for col in feature_cols:
        val1 = str(row1[col]).strip()
        val2 = str(row2[col]).strip()
        if val1 == "NA" and val2 == "NA":
            continue
        match = val1 == val2
        results.append((col, val1, val2, match))
    num_matches = sum(r[3] for r in results)
    total_compared = len(results)
    return num_matches, total_compared, results


def sign_differences(row1, row2, feature_cols):
    """
    Returns the number of differences and total compared features (ignoring both-NA pairs)
    and a list of feature-wise comparison (feature, row1_value, row2_value, match).
    """
    results = []
    for col in feature_cols:
        val1 = str(row1[col]).strip()
        val2 = str(row2[col]).strip()
        if val1 == "NA" and val2 == "NA":
            continue
        match = val1 != val2
        results.append((col, val1, val2, match))
    num_matches = sum(r[3] for r in results)
    total_compared = len(results)
    return num_matches, total_compared, results


def vector_of_differences(input_row, df, feature_cols):
    """
    Returns a vector of differences (i.e., number of differing features) between
    input_row and each row of df, following the same NA logic.
    """
    diffs = []
    for idx, row in df.iterrows():
        _, total_compared, results = sign_similarity(input_row, row, feature_cols)
        num_diffs = total_compared - sum(r[3] for r in results)
        diffs.append(num_diffs)
    return diffs


def number_of_x(input_row, df, feature_cols, bool_fn):
    """
    Returns the number of rows in df for which bool_fn(num_diffs) is True.
    """
    count = 0
    for idx, row in df.iterrows():
        _, total_compared, results = sign_similarity(input_row, row, feature_cols)
        num_diffs = total_compared - sum(r[3] for r in results)
        if bool_fn(num_diffs):
            count += 1
    return count



if __name__ == "__main__":
    file_path = "data/signdata.csv"

    # List of features to compare
    feature_cols = [
        "SelectedFingers.2.0", "Flexion.2.0", "FlexionChange.2.0", "Spread.2.0", "SpreadChange.2.0",
        "ThumbPosition.2.0", "ThumbContact.2.0", "SignType.2.0", "Movement.2.0", "RepeatedMovement.2.0",
        "MajorLocation.2.0", "MinorLocation.2.0", "SecondMinorLocation.2.0", "Contact.2.0",
        "NonDominantHandshape.2.0", "UlnarRotation.2.0"
    ]

    # Load dataset
    df = load_sign_dataset(file_path)

    # Demo: Compare two rows (e.g., rows 2 and 3)
    idx1, idx2 = 2394, 269
    num_matches, total_compared, results = sign_similarity(df.iloc[idx1], df.iloc[idx2], feature_cols)
    num_differences, total_compared, results = sign_differences(df.iloc[idx1], df.iloc[idx2], feature_cols)

    print(
        f"Comparing row {idx1} (LemmaID={df.iloc[idx1]['LemmaID']}) and row {idx2} (LemmaID={df.iloc[idx2]['LemmaID']}):")
    print(f"Matches: {num_matches}/{total_compared}")
    print(f"Distance: {num_differences}/{total_compared}")
    for feature, val1, val2, match in results:
        print(f"{feature}: {val1} vs {val2} --> {'Match' if match else 'Different'}")

    print("\nVector of differences from input row to every row in the dataset:")
    input_row = df.iloc[idx1]
    diff_vector = vector_of_differences(input_row, df, feature_cols)
    print(diff_vector)

    n_exact_matches = number_of_x(input_row, df, feature_cols, lambda x: x == 0)
    print(f"Number of exact matches: {n_exact_matches}")

    n_within_2 = number_of_x(input_row, df, feature_cols, lambda x: x <= 2)
    print(f"Number of signs within 2 differences: {n_within_2}")

