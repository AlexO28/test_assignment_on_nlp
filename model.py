# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:12:14 2023

@author: Семья
"""
import pandas as pd
from processing import (
    y_dict,
    decode_to_target,
    decode_to_true_date,
    add_columns_for_catboost,
)
from lemmatization import prepare_lemmatized_text, remove_punctuation
from catboost import CatBoostClassifier


def main(filename):
    tab = pd.read_csv("file.csv", sep=";")
    if 'date' not in tab.columns:
        tab['date'] = "2021-02-15"
    tab["true_date"] = decode_to_true_date(tab)
    tab["text_employer"] = remove_punctuation(tab)
    tab["text"] = tab["text_employer"].apply(lambda x: prepare_lemmatized_text(x))
    words = pd.read_csv("words.csv")
    words = words["words"].tolist()
    y = tab["ACTION_ITEM_RESULT_PRODUCT_NAME"]
    tab = tab[["text", "true_date"]]
    tab = add_columns_for_catboost(tab, words)
    del tab['text']
    y = y.apply(lambda x: y_dict[x])
    model = CatBoostClassifier()
    model.load_model("model_catboost")
    y_pred = model.predict(tab)
    y_pred = [y[0] for y in y_pred]
    tab = pd.read_csv("file.csv", sep=";")
    tab["prediction"] = y_pred
    y_target = decode_to_target(tab)
    print("Accuracy is ", len(y_target[y_target == y_pred]) / len(y_target))
    tab.to_csv("results.csv")


if __name__ == "__main__":
    main("file.csv")
