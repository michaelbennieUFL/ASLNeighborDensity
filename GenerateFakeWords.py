import numbers
from typing import List, Tuple

import os
import pandas as pd

from DistanceMatrix import load_distance_matrix, extract_close_pairs

from tqdm import tqdm


def get_phonemic_data_tuple(row:List[str])->Tuple[str]:
    return tuple(row[2:])


def loadPossibleCategories(categoriesFile:str)->List[List[str]]:
    categories = []
    with open(categoriesFile) as f:
        f.readline()
        for line in f:
            categories.append(line.split(",")[2].replace("\n","").split(sep=";"))
    return categories



def check_if_different_and_not_nan(idx, first_series, second_series) -> bool:
    """
    在 idx 位置上：
      - 先用 .iloc 取出两个值 v1, v2
      - 如果任一是数字（int/float），就返回 False（跳过这对）
      - 否则，比较它们是否不相等，返回 True/False
    """
    v1 = first_series.iloc[idx]
    v2 = second_series.iloc[idx]

    # 如果是個nan，就“跳过”
    if pd.isna(v1) or pd.isna(v2):
        return pd.isna(v1) != pd.isna(v2)

    # 此时两者都不是nan，用常规不等判断
    return v1 != v2


def generatePossibleFakeWords(firstPhonemeicTranscription:List[str],secondPhonemeicTranscription:List[str],phonemicDistance, categories)->List[List[str]]:
    possibleFakeWords = []

    if phonemicDistance == 1:
        common_index:int=2
        while common_index < len(firstPhonemeicTranscription):
            if check_if_different_and_not_nan(common_index,firstPhonemeicTranscription,secondPhonemeicTranscription) and len(categories[common_index-2])>2:
                for phoneme in categories[common_index-2]:
                    if (phoneme == str(firstPhonemeicTranscription.iloc[common_index])
                            or str(phoneme == secondPhonemeicTranscription.iloc[common_index])):
                        continue
                    new_fake_word = firstPhonemeicTranscription.copy()
                    new_fake_word.iloc[0] += "*"
                    new_fake_word.iloc[common_index] = phoneme
                    possibleFakeWords.append(new_fake_word)
            common_index+=1
    elif phonemicDistance == 2:
        unequal_indicies=[]
        common_index:int=2
        while common_index < len(firstPhonemeicTranscription):
            if check_if_different_and_not_nan(common_index,firstPhonemeicTranscription,secondPhonemeicTranscription):
                unequal_indicies.append(common_index)
            common_index+=1
        for index in range(2,len(firstPhonemeicTranscription)):
            if index not in unequal_indicies:
                for phoneme in categories[index-2]:
                    new_fake_word = firstPhonemeicTranscription.copy()
                    new_fake_word.iloc[0]+="*"
                    new_fake_word.iloc[index] = phoneme
                    possibleFakeWords.append(new_fake_word)


    return possibleFakeWords





def generate_and_save_fake_words(
    distance_matrix_file: str,
    categories_file: str,
    phonemic_df_file: str,
    output_file: str
):


    # —— 1. 读入距离矩阵，提取所有距离为1或2的配对 ——
    dist_mat = load_distance_matrix(distance_matrix_file)
    close_pairs = extract_close_pairs(dist_mat, distances=(1, 2))


    # —— 2. 读入类别列表 ——
    categories = loadPossibleCategories(categories_file)

    # —— 3. 读入一个包含“索引 ↔ 音素转写”映射的表格 ——
    # 假设 phonemic_df_file 是一个 CSV，里面有两列：index（或使用 DataFrame 的行号）和
    # PhonemicTranscription（空格分隔的音素列表）
    phon_df = pd.read_csv(phonemic_df_file, index_col=0, encoding='utf-8')


    real_phonemes= {get_phonemic_data_tuple(list(row)) for _, row in phon_df.iterrows()}

    fake_phonemes = {}

    # —— 4. 为每一对生成假词并写入文件 ——
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    for _, row in tqdm(close_pairs.iterrows()):
        i, j, dist = row['index1'], row['index2'], row['distance']
        # 从 phon_df 中取出音素列表
        first = phon_df.iloc[int(i)]
        second = phon_df.iloc[int(j)]
        # 生成所有假词
        fakes = generatePossibleFakeWords(first, second, dist, categories)
        # 写入每一个假词
        for fake in fakes:
            fale_phonemic_data=get_phonemic_data_tuple(fake)
            phonmeic_code=[first["LemmaID"]+";"+first["Code"],second["LemmaID"]+";"+second["Code"]]
            if fale_phonemic_data in real_phonemes:
                #print(f"Fake word {fale_phonemic_data} already exists in real phonemes")
                continue
            elif fale_phonemic_data in fake_phonemes:
                fake_phonemes[fale_phonemic_data].update(phonmeic_code)
                continue
            else:
                fake_phonemes[fale_phonemic_data]=set(phonmeic_code)

    feature_names = [
        "SelectedFingers.2.0", "Flexion.2.0", "FlexionChange.2.0", "Spread.2.0",
        "SpreadChange.2.0", "ThumbPosition.2.0", "ThumbContact.2.0", "SignType.2.0",
        "Movement.2.0", "RepeatedMovement.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
        "SecondMinorLocation.2.0", "Contact.2.0", "NonDominantHandshape.2.0", "UlnarRotation.2.0"
    ]

    with open(output_file, 'w', encoding='utf-8') as fout:
        # 寫入 header
        headers = ["neighbors[list]"] + feature_names
        fout.write("\t".join(headers) + "\n")

        # 逐筆寫入
        for phon_tuple, code_set in fake_phonemes.items():
            # 第一欄：把最近真實詞的 code set 轉成 [code1,code2,...] 的字串
            neighbors_str = "[" + ",".join(code_set) + "]"

            # 後面各欄：phon_tuple 本身就是對應 feature_names 的值序列
            feature_vals = [str(val) for val in phon_tuple]

            # 合併並寫到檔案
            line = "\t".join([neighbors_str] + feature_vals)
            fout.write(line + "\n")

    print(f"All fake words saved to {output_file} "
          f"(total pairs: {len(fake_phonemes.items())})")

if __name__ == "__main__":
    categories = loadPossibleCategories("./data/Unique_Counts_and_Values_per_Column.csv")
    print(categories)

    generate_and_save_fake_words(
        distance_matrix_file="data/distanceMatrix.txt",
        categories_file="data/Unique_Counts_and_Values_per_Column.csv",
        phonemic_df_file="data/signdata_slimmed.csv",
        output_file="data/fake_words.tsv"
    )

