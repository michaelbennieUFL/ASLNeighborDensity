import pandas as pd
import CalcuateDistance as cd

def validate_fake_neighbors(
    fake_file: str,
    sign_dataset_file: str,
    feature_cols: list = None
) -> pd.DataFrame:
    """
    驗證每個假詞的「距離＝1 鄰居」是否和 TSV 中存的相符，
    並檢查假詞的特徵組合是否已存在於真實手語資料集中。

    參數:
      fake_file: 先前產生的假詞 TSV 檔路徑（必須包含 'neighbors[list]' 欄位）。
      sign_dataset_file: 真實手語資料集路徑（CSV，包含 LemmaID, Code 及所有 feature）。
      feature_cols: 欲比對的 feature 欄位清單；若為 None，會用 CalculateDistance 裡的預設欄位。
    回傳:
      pd.DataFrame，每一列對應假詞的一筆驗證結果。
    """
    # 1. 讀真實資料集
    df_real = cd.load_sign_dataset(sign_dataset_file)

    # 2. feature 欄位預設
    if feature_cols is None:
        feature_cols = [
            "SelectedFingers.2.0", "Flexion.2.0", "FlexionChange.2.0",
            "Spread.2.0", "SpreadChange.2.0", "ThumbPosition.2.0",
            "ThumbContact.2.0", "SignType.2.0", "Movement.2.0",
            "RepeatedMovement.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
            "SecondMinorLocation.2.0", "Contact.2.0",
            "NonDominantHandshape.2.0", "UlnarRotation.2.0"
        ]

    # 3. 讀假詞 TSV
    fake_df = pd.read_csv(fake_file, sep='\t', encoding='utf-8')

    # 解析 neighbors[list] 欄位
    def parse_neighbors(neigh_str: str) -> list:
        s = neigh_str.strip()[1:-1]
        return s.split(',') if s else []

    records = []
    for idx, row in fake_df.iterrows():
        # 特徵向量
        input_feats = row[feature_cols].fillna("NA").astype(str)
        # 計算與真實資料集中每列的差異數
        diffs = cd.vector_of_differences(input_feats, df_real, feature_cols)

        # 找出所有差異=1 的真實詞索引
        neighbor_idxs = [i for i, d in enumerate(diffs) if d == 1]
        # 由 idx 取出 LemmaID;Code
        computed_codes = {
            f"{df_real.iloc[i]['LemmaID']};{df_real.iloc[i]['Code']}"
            for i in neighbor_idxs
        }

        # TSV 裡存的 neighbors
        stored_codes = set(parse_neighbors(row["neighbors[list]"]))

        # 是否有在真實裡「完全相同」（diff=0）
        duplicate = any(d == 0 for d in diffs)

        records.append({
            "fake_index": idx,
            "neighbor_count": len(computed_codes),
            "stored_count":    len(stored_codes),
            "counts_match":    (len(computed_codes) == len(stored_codes)),
            "neighbors_match": (computed_codes == stored_codes),
            "duplicate_in_real": duplicate
        })

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    df_check = validate_fake_neighbors(
        fake_file="data/fake_words.tsv",
        sign_dataset_file="data/signdata_slimmed.csv"
    )
    print(df_check)
