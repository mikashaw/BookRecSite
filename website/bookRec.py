import tensorflow as tf
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle

ratings_df = pd.read_csv('/Users/mikashaw/code/ML_Projects/BookRecommender/website/ratings.csv', encoding = 'utf-8')
books_df = pd.read_csv('/Users/mikashaw/code/ML_Projects/BookRecommender/website/books.csv', encoding = 'utf-8')

Xtrain, Xtest = train_test_split(ratings_df, test_size = 0.2, random_state = 1)

nbook_id = ratings_df.book_id.nunique()
nuser_id = ratings_df.user_id.nunique()

# book input network

input_books = tf.keras.layers.Input(shape=[1])
embed_books = tf.keras.layers.Embedding(nbook_id + 1, 15)(input_books)
output_books = tf.keras.layers.Flatten()(embed_books)

# user input network

input_user = tf.keras.layers.Input(shape=[1])
embed_user = tf.keras.layers.Embedding(nuser_id+1, 15)(input_user)
output_user = tf.keras.layers.Flatten()(embed_user)

# concatenated network

concat_layer = tf.keras.layers.Concatenate()([output_books,output_user])
x = tf.keras.layers.Dense(128, activation = 'relu')(concat_layer)
x_out = tf.keras.layers.Dense(1, activation = 'relu')(x)

model = tf.keras.Model(inputs = [input_books, input_user], outputs = x_out)

# Compiling the model

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'mse', metrics = ['acc'])

model_history = model.fit([Xtrain.book_id, Xtrain.user_id], Xtrain.rating, batch_size = 64, epochs = 5, verbose = 1, validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating))

print("model successfully compliled")

#mika, get rid of this and used a tensorflow compatible way to save the model...
model.save("savedModel/myModel")
