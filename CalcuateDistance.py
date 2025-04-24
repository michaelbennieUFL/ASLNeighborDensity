import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from tqdm import tqdm
import os
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

def checkNaNString(text:str)->bool:
    return text.lower()=="nan" or text=="NA"

def sign_similarity(row1, row2, feature_cols):
    """
    Returns the number of matches and total compared features (ignoring both-NA pairs)
    and a list of feature-wise comparison (feature, row1_value, row2_value, match).
    """
    results = []
    for col in feature_cols:
        val1 = str(row1[col]).strip()
        val2 = str(row2[col]).strip()
        if checkNaNString(val1) and checkNaNString(val2):
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
        if checkNaNString(val1) and checkNaNString(val2):
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


def _init_pool(df_, feature_cols_):
    # share these in each worker so we don't pickle them for every task
    global _df, _feature_cols, _n
    _df = df_
    _feature_cols = feature_cols_
    _n = len(_df)


def _compute_row(i):
    """
    For a given i, compute (i, j, diff) for all j > i.
    """
    row_i = _df.iloc[i][_feature_cols].fillna("NA").astype(str)
    out = []
    for j in range(i + 1, _n):
        row_j = _df.iloc[j][_feature_cols].fillna("NA").astype(str)
        both_na = (row_i == "NA") & (row_j == "NA")
        diff = int(((row_i != row_j) & ~both_na).sum())
        out.append((i, j, diff))
    return out


def distance_matrix_upper_parallel(df, feature_cols, n_jobs=24):
    """
    Parallel upper‐triangular distance matrix using n_jobs workers.
    """
    n = len(df)
    idx = df.index
    dist_mat = pd.DataFrame(index=idx, columns=idx, dtype="Int64")

    # set diagonal = 0
    diag_ix = np.arange(n)
    dist_mat.values[diag_ix, diag_ix] = 0

    # launch pool, each worker gets a reference to df and feature_cols
    with mp.Pool(processes=n_jobs, initializer=_init_pool, initargs=(df, feature_cols)) as pool:
        # map i↦row‐results in parallel
        for row_results in tqdm(pool.imap(_compute_row, range(n)), total=n):
            for i, j, diff in row_results:
                dist_mat.iat[i, j] = diff

    return dist_mat

if __name__ == "__main__":
    file_path = "data/signdata_slimmed.csv"

    # List of features to compare
    feature_cols = [
        "SelectedFingers.2.0", "Flexion.2.0", "FlexionChange.2.0", "Spread.2.0", "SpreadChange.2.0",
        "ThumbPosition.2.0", "ThumbContact.2.0", "SignType.2.0", "Movement.2.0", "RepeatedMovement.2.0",
        "MajorLocation.2.0", "MinorLocation.2.0", "SecondMinorLocation.2.0", "Contact.2.0",
        "NonDominantHandshape.2.0", "UlnarRotation.2.0"
    ]

    # Load dataset
    df = load_sign_dataset(file_path)

    raw = """SelectedFingers.2.0,i
Flexion.2.0,FullyOpen
FlexionChange.2.0,0.0
Spread.2.0,1.0
SpreadChange.2.0,0.0
ThumbPosition.2.0,Open
ThumbContact.2.0,0.0
SignType.2.0,OneHanded
Movement.2.0,NA
RepeatedMovement.2.0,1
MajorLocation.2.0,Neutral
MinorLocation.2.0,Neutral
SecondMinorLocation.2.0,NA
Contact.2.0,0
NonDominantHandshape.2.0,NA
UlnarRotation.2.0,1"""

    d = {line.split(',', 1)[0]: line.split(',', 1)[1]
         for line in raw.splitlines()}

    s = pd.Series(d, name="value")

    idx1 = 1054
    num_matches, total_compared, results = sign_similarity(df.iloc[idx1], s, feature_cols)
    num_differences, total_compared, results = sign_differences(df.iloc[idx1], s, feature_cols)

    print(
        f"Comparing row {idx1} (LemmaID={df.iloc[idx1]['LemmaID']}) and fake word):")
    print(f"Matches: {num_matches}/{total_compared}")
    print(f"Distance: {num_differences}/{total_compared}")

    for feature, val1, val2, match in results:
        print(f"{feature}: {val1} vs {val2} --> {'Match' if not match else 'Different'}")

    # Demo: Compare two rows (e.g., rows 2 and 3)
    idx1, idx2 = 0, 1052
    num_matches, total_compared, results = sign_similarity(df.iloc[idx1], df.iloc[idx2], feature_cols)
    num_differences, total_compared, results = sign_differences(df.iloc[idx1], df.iloc[idx2], feature_cols)

    print(
        f"Comparing row {idx1} (LemmaID={df.iloc[idx1]['LemmaID']}) and row {idx2} (LemmaID={df.iloc[idx2]['LemmaID']}):")
    print(f"Matches: {num_matches}/{total_compared}")
    print(f"Distance: {num_differences}/{total_compared}")
    for feature, val1, val2, match in results:
        print(f"{feature}: {val1} vs {val2} --> {'Match' if not match else 'Different'}")

    print("\nVector of differences from input row to every row in the dataset:")
    input_row = df.iloc[idx1]
    diff_vector = vector_of_differences(input_row, df, feature_cols)
    print(diff_vector)

    n_exact_matches = number_of_x(input_row, df, feature_cols, lambda x: x == 0)
    print(f"Number of exact matches: {n_exact_matches}")


    # exit()
    #
    # dist_full = distance_matrix_upper_parallel(df, feature_cols)
    # os.makedirs("data", exist_ok=True)
    #
    # # write as a tab-separated text file (with row and column labels)
    # output_path = "data/distanceMatrix.txt"
    # dist_full.to_csv(output_path, sep="\t", na_rep="NA")
    # print(f"Distance matrix saved to {output_path}")
    #
    # print(dist_full)



