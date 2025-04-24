import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_distance_matrix(filepath):
    """
    Load a tab-separated upper-triangular distance matrix from a text file.
    Expects row labels in the first column and tabs as separators.
    Returns a pandas DataFrame with Int64 dtype.
    """
    df = pd.read_csv(filepath, sep="\t", index_col=0, dtype="Int64")
    return df


def plot_distance_distribution(dist_mat):
    """
    Plot the distribution of strictly upper-triangle values of the distance matrix.
    """
    mask = np.triu(np.ones(dist_mat.shape, dtype=bool), k=1)
    upper_vals = dist_mat.values[mask]
    upper_vals = upper_vals[~pd.isna(upper_vals)].astype(int)

    plt.figure()
    plt.hist(upper_vals,
             bins=range(upper_vals.min(), upper_vals.max() + 2),
             edgecolor='black')
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pairwise Distances')
    plt.xticks(range(upper_vals.min(), upper_vals.max() + 1))
    plt.show()

def extract_close_pairs(dist_mat, distances=(1, 2)):
    """
    从上三角（k=1，不含对角线）提取所有 (index1, index2, distance) 对，
    并只保留距离在 distances 中的配对。
    """
    # 1. 构造上三角掩码（不包括对角线）
    mask = np.triu(np.ones(dist_mat.shape, dtype=bool), k=1)
    # 2. 只保留上三角的数据，其他都变成 NA
    upper = dist_mat.where(mask)
    # 3. 把 DataFrame 扁平化成三列（行索引、列索引、值）
    flat = upper.stack().reset_index()
    flat.columns = ['index1', 'index2', 'distance']
    # 4. 过滤出 distance 等于 1 或 2 的
    return flat[flat['distance'].isin(distances)].reset_index(drop=True)


if __name__ == "__main__":
    dist_mat = load_distance_matrix("data/distanceMatrix.txt")
    close_pairs = extract_close_pairs(dist_mat, distances=(1, 2))
    print(close_pairs)
    print(close_pairs.shape)
    plot_distance_distribution(dist_mat)
