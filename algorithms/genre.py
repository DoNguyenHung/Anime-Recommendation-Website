from lenskit.algorithms import Predictor
import numpy as np
import pandas as pd

class GenreRec(Predictor):
    def fit(self, ratings, genre):
        row_norms = np.linalg.norm(genre, axis=1, keepdims=True)
        self.genre_ = genre / row_norms
        self.ratings_ = ratings[["user", "rating", "item"]].astype({"user": "int32", "rating": "float32", "item": "int32"})
    
    def predict_for_user(self, user, items=None, ratings=None):
        if ratings is None:
            ratings = self.ratings_

        user_ratings = self.ratings_[(self.ratings_['user'] == user) & self.ratings_['item'].isin(self.genre_.index)]
        user_items = self.genre_.loc[user_ratings['item']]
        scores = user_items.mul(user_ratings.set_index("item")["rating"], axis=0).mean() @ self.genre_.T
        return scores
