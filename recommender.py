import numpy as numpy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating = 3.8)

print(repr(data['train']))
print(repr(data['test']))

algo = LightFM(loss = 'warp')
algo.fit(data['train'], epochs=100, num_threads=2)

def sample_recommendation(algo, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        liked_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = algo.predict(user_id, numpy.arange(n_items))
        top_items = data['item_labels'][numpy.argsort(-scores)]
        print("User %s" % user_id)
        print("     Liked Movies:")

        for x in liked_movies[:10]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:10]:
            print("        %s" % x)
            
sample_recommendation(algo, data, [3, 25, 451])



