import numpy as np
import keras as K
from sklearn.model_selection import train_test_split

model = K.models.load_model("./models/sentanalysis.h5")

model.summary()

index = K.datasets.imdb.get_word_index()
review = "this movie was great"


words = review.split()
review = []
for word in words:
    if word in index and index[word] < 1997:
        review.append(index[word] + 3)
    else:
        review.append(2)
review = K.preprocessing.sequence.pad_sequences([review])
prediction = model.predict(review)
print("prediction is: ", prediction[0][0])