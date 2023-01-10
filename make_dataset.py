"""
プログラムの説明：
DB wikipedia_FAに保存されているすべてのテーブルに対して、ja_docvecとen_docvecのペアをローカルへ持ってきてpklファイルにまとめる。
ただし、日本語のデータが無い行は無視する。
総データ数は2632, 内8割を訓練用データ、残りをテスト用データとする。

"""
import pickle
import numpy as np
from modules.dbManager import DBManager

DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_NAME = "wikipedia_featured_articles"

db_manager = DBManager(hostname=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)


def main():
    # all_dataからトレーニング用とテスト用に分割して格納する
    table_names = fetch_tablenames()
    en_vecs, ja_vecs = get_all_vectors(table_names)
    # print(en_vecs[0:2])
    # print(len(en_vecs))
    # print(ja_vecs[0:2])
    # print(len(ja_vecs))
    all_size = len(en_vecs)
    train_size = int(all_size * 0.8)

    x_train = en_vecs[0:train_size]
    t_train = ja_vecs[0:train_size]
    x_test = en_vecs[train_size:]
    t_test = ja_vecs[train_size:]
    print("x_train", len(x_train))
    print("t_train", len(t_train))
    print("x_test", len(x_test))
    print("t_test", len(t_test))

    # pklファイルにデータを保存
    with open("dataset/dataset.pkl", "wb") as f:
        pickle.dump((x_train, t_train, x_test, t_test),  f)


def get_all_vectors(table_names):
    """
    DBから取ってきた英語と日本語の文書ベクトルをndarrayに変換して返す
    """
    # N x 100の形で一時的に日本語と英語のベクトルを格納する
    en_vecs = []
    ja_vecs = []
    for table_name in table_names:
        query = f"SELECT en_docvec, ja_docvec FROM {table_name} WHERE ja_docvec != '{{}}' ORDER BY page_id ASC"
        enja_pair_list = db_manager.select(query)

        for enja_pair in enja_pair_list:
            en_vec = enja_pair[0]
            ja_vec = enja_pair[1]
            if en_vec == [] or ja_vec == []:
                print("exception: data is None")
                continue
            en_vecs.append(en_vec)
            ja_vecs.append(ja_vec)
    en_vecs = np.array(en_vecs)
    ja_vecs = np.array(ja_vecs)
    return en_vecs, ja_vecs


def fetch_tablenames():
    table_names = db_manager.fetch_tablenames()
    table_names = [table_name[0] for table_name in table_names]
    table_names.sort()  # 辞書順にソート
    return table_names


if __name__ == "__main__":
    main()
