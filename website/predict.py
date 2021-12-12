import tensorflow as tf 
import numpy as np
import pandas as pd 


def load_data():
    books_df= pd.read_csv("Users/mikashaw/code/ML_Projects/BookRecommender/website/books.csv")
    ratings_df = pd.read_csv("Users/mikashaw/code/ML_Projects/BookRecommender/website/ratings.csv")
    return (books_df, ratings_df)

def load_model():
    model = tf.keras.models.load_model("savedModel/myModel")
    return model


def predict(user):

    books_df, ratings_df = load_data()
    model = load_model()
    books_df_copy = books_df.copy()
    books_df_copy = books_df_copy.set_index("book_id")

    b_id = list(ratings_df.book_id.unique())
    book_arr = np.array(b_id)
    user = np.array([user.id for i in range(len(b_id))])

    pred = model.predict([book_arr, user])

    web_book_data = postProcess(pred)

    ordered_data = order_data(web_book_data)
    return ordered_data


def postProcess(pred, books_df):
    pred = pred.reshape(-1)
    pred_ids = (-pred).argsort()[0:5]
    web_book_data = books_df.iloc[pred_ids]
    return web_book_data

def order_data(web_book_data):
    data = {}
    for row in range(len(web_book_data)):
        data[row] = {'title': row.title, 'image_url': row.image_url, 'author': row.authors}
    return data