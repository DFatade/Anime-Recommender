import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors


link='anime_cleaned.csv'
data=pd.read_csv(link)

def preprocess_data():
    link='anime_cleaned.csv'
    data=pd.read_csv(link)
    type_encoded=pd.get_dummies(data['type'])
    source_encoded=pd.get_dummies(data['source'])
    genre_encoded=data['genre'].str.get_dummies(sep=',')
    features=pd.concat([genre_encoded, type_encoded,data['episodes']],axis=1)
    features_scaled=MinMaxScaler().fit_transform(features)
    return features_scaled

def get_partial_names(title):
    names=list(data.title.values)
    for name in names:
        if title in name:
            return [name, names.index(name)]


def get_features(title):
    values=get_partial_names(title)
    return values[1]

def get_vector(title):
    index=get_features(title)
    data=preprocess_data()
    return data[index]

def collaborative_filter():
    data=preprocess_data()
    filtering=NearestNeighbors().fit(data)
    return filtering


def get_recommendations(title):
    vectorized_input=get_vector(title)
    filter_model=collaborative_filter()
    indices=filter_model.kneighbors([vectorized_input])[1]
    recommendations=data['title'].iloc[indices[0],].values
    return recommendations
print(get_recommendations("One Piece"))
