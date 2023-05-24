# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:46:59 2023

@author: Семья
"""
import pandas as pd
from datetime import datetime
from stop_words import get_stop_words


y_dict = {
    "Зарплатные проекты": 2,
    "Бизнес-карта": 1,
    "Эквайринг": 4,
    "Открытие банковского счета": 3,
}

stop_words = get_stop_words("ru")


def decode_to_target(tab):
    return tab["ACTION_ITEM_RESULT_PRODUCT_NAME"].apply(lambda x: y_dict[x])


def decode_to_true_date(tab):
    date_min = tab["date"].min()
    return tab["date"].apply(
        lambda x: (
            datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(date_min, "%Y-%m-%d")
        ).days
    )


def get_aggregated_statistics_on_words_for_category(tab, category):
    tab_red = tab[tab["ACTION_ITEM_RESULT_PRODUCT_NAME"] == category]
    large_text = " ".join(tab_red["text"].tolist())
    large_text = " ".join(
        [word for word in large_text.split(" ") if word not in stop_words]
    )
    aggtab = pd.DataFrame(large_text.split(" ")).value_counts()
    return aggtab


def add_columns_for_catboost(tab, words):
    for word in words:
        try:
            tab[word] = tab["text"].apply(lambda x: word in x.split(" "))
            tab[word] = tab[word].astype(int)
        except:
            try:
                tab[word] = 0
            except:
                print("miss")
    return tab
