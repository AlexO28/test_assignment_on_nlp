# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:49:59 2023

@author: Семья
"""
import re
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


def prepare_lemmatized_text(text_string):
    doc = Doc(text_string)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    lemma_text = {_.text: _.lemma for _ in doc.tokens}
    words = text_string.split(" ")
    newline = []
    for word in words:
        try:
            newline.append(lemma_text[word])
        except:
            newline.append(word)
    return " ".join(newline)


def remove_punctuation(tab):
    return tab["text_employer"].apply(lambda x: re.sub(r"[^\w\s]", " ", x))

