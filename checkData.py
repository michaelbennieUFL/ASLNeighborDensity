import pandas as pd
import CalcuateDistance as cd
from tqdm.contrib.concurrent import process_map

# 預設要比較的 feature 欄位
FEATURE_COLS = [
    "SelectedFingers.2.0", "Flexion.2.0", "FlexionChange.2.0",
    "Spread.2.0", "SpreadChange.2.0", "ThumbPosition.2.0",
    "ThumbContact.2.0", "SignType.2.0", "Movement.2.0",
    "RepeatedMovement.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
    "SecondMinorLocation.2.0", "Contact.2.0",
    "NonDominantHandshape.2.0", "UlnarRotation.2.0"
]

def parse_neighbors(neigh_str: str) -> list:
    """
    將 TSV 欄位形式的 "[L1;C1,L2;C2,...]" 轉成 list of "LemmaID;Code"
    """
    s = neigh_str.strip()[1:-1]
    return s.split(',') if s else []

def process_item(item):
    """
    處理單筆 fake_df.iterrows() 傳來的 (idx, row)，回傳一筆 record dict。
    會使用全域變數 df_real, FEATURE_COLS。
    """
    idx, row = item

    input_feats = row[FEATURE_COLS].fillna("NA").astype(str)
    # 計算與真實資料集每列的差異數
    diffs = cd.vector_of_differences(input_feats, df_real, FEATURE_COLS)


    neighbor_idxs = [i for i, d in enumerate(diffs) if d == 1]
    computed_codes = {
        f"{df_real.iloc[i]['LemmaID']};{df_real.iloc[i]['Code']}"
        for i in neighbor_idxs
    }

    stored_codes = set(parse_neighbors(row["neighbors[list]"]))

    duplicate = any(d == 0 for d in diffs)

    if duplicate:
        print(f"WARNING: Fake word {idx} has exact matches!")

    if  not all(x in computed_codes for x in stored_codes):
        print(f"FAILURE: Fake word {idx}:")
        print(f"  computed_codes: {computed_codes}")
        print(f"  stored_codes:   {stored_codes}")
        print(f"  duplicate:      {duplicate}")
        print()

    return {
        "fake_index": idx,
        "neighbor_count":     len(computed_codes),
        "stored_count":       len(stored_codes),
        "counts_match":       (len(computed_codes) == len(stored_codes)),
        "neighbors_match":    (computed_codes == stored_codes),
        "neighbors": computed_codes,
        "duplicate_in_real":  duplicate
    }

if __name__ == "__main__":
    # 檔案路徑
    fake_file = "data/fake_words.tsv"
    sign_dataset_file = "data/signdata_slimmed.csv"

    # 1. 讀真實手語資料
    df_real = cd.load_sign_dataset(sign_dataset_file)

    # 2. 讀假詞 TSV
    fake_df = pd.read_csv(fake_file, sep='\t', encoding='utf-8')

    # 3. 平行處理 + tqdm 顯示進度
    # total 設為 384263，max_workers 設為 16
    records = process_map(
        process_item,
        fake_df.iterrows(),
        max_workers=22,
        total=6013
    )


    # 4. 組成 DataFrame 並 merge
    df_check = pd.DataFrame.from_records(records).set_index("fake_index")
    df_merged = fake_df.join(df_check, how="inner")

    # 過濾
    sel = (~df_merged["duplicate_in_real"]) & (df_merged["stored_count"] >= 2)
    df_sel = df_merged.loc[sel].copy()

    # **重要：把 neighbors[list] 換成我們新算出來的那欄 neighbors**
    df_sel["neighbors[list]"] = df_sel["neighbors"].apply(
        lambda s: "[" + ",".join(sorted(s)) + "]"
    )

    # 準備要輸出的欄位順序
    out_cols = ["neighbor_count", "neighbors[list]"] + FEATURE_COLS

    # 輸出 TSV
    df_sel.to_csv(
        "data/filtered_fake_words.tsv",
        sep="\t",
        columns=out_cols,
        index=False,
        encoding="utf-8"
    )
    print(f"Saved {len(df_sel)} rows to filtered_fake_words.tsv")


