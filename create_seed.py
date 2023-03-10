"""Generate a SQL script to seed the SQLite database"""

import numpy as np
import pandas as pd
import random
from prisma import Prisma
from scipy.interpolate import interp1d

random.seed(0)
np.random.seed(0)

df = pd.read_csv('./data/imdb_top_1000.csv')
df = df.loc[:,['Series_Title', 'Runtime', 'Genre', 'IMDB_Rating']]
df['Runtime'] = df['Runtime'].apply(lambda d: int(d.replace('min', '').replace(' ', '')))

genres = [d.split(',') for d in df['Genre'].tolist()]
genres = set([item.strip() for sublist in genres for item in sublist])

genre_distributions = {}
for genre in genres:
    mu_map = interp1d([0,1], [10, 22])
    sigma_map = interp1d([0,1], [1,5])
    genre_distributions[genre] = {
        'mu': int(mu_map(random.random())),
        'sigma': sigma_map(random.random())
    }

genre_distributions['Animation'] = {'mu': 10, 'sigma': 2}
genre_distributions['Family'] = {'mu': 10, 'sigma': 4}

# Write genres
sql = '''
delete from _FilmToGenre;
delete from GenreGaussian;
delete from Genre;
delete from Screen;
delete from Cinema;
delete from Film;

insert or ignore into Cinema (id, name, opens, closes, "interval", cleaningTime)
values (1, "ABC Queen Street", 8, 23, 15, 20);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (1, 'Screen 1', 200, 1);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (2, 'Screen 2', 300, 1);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (3, 'Screen 3', 100, 1);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (4, 'Screen 4', 150, 1);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (5, 'Screen 5', 120, 1);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (6, 'Screen 6', 200, 1);

insert or ignore into Screen (id, name, capacity, cinemaId)
values (7, 'Screen 7', 250, 1);

'''

dist_id = 1
for i, key in enumerate(genre_distributions):
    mu, sigma = genre_distributions[key]['mu'], genre_distributions[key]['sigma']
    weight = 1
    genre_distributions[key]['id'] = i + 1
    sql += f'insert or ignore into Genre (id, name) values ({i + 1}, "{key}");\n'
    sql += f'insert or ignore into GenreGaussian (id, genreId, mu, sigma, weight) values ({dist_id}, {i + 1}, {mu}, {sigma}, {weight});\n'
    dist_id += 1

sql += '\n'
for index, row in df.iterrows():
    sql += f'insert or ignore into Film (id, title, runtime, rating) values ({index + 1}, "{row["Series_Title"]}", {row["Runtime"]}, {row["IMDB_Rating"]});\n'
    genres = [d.strip() for d in row['Genre'].split(',')]
    for genre in genres:
        genre_id = genre_distributions[genre]['id']
        sql += f'insert or ignore into _FilmToGenre (A, B) values ({index + 1}, {genre_id});\n'

with open('./data/seed.sql','w') as f:
    f.write(sql)
