"""Generate training data using the agent"""

import os
import timeit
from csv import DictWriter

import numpy as np
from super_hash import super_hash

from app.dependencies.caching import MemoryCache
from app.dependencies.data import (CinemaService, Database,
                                   DistributionService, FilmService,
                                   GenreService, ScreenService)
from models.agents import Agent
from models.utils import get_film_encoding, get_screen_encoding

if __name__ == "__main__":
    print("Film Scheduler - Data Generation")

    cache = MemoryCache()
    database = Database()
    cinema_service = CinemaService(database, cache)
    screen_service = ScreenService(database, cache)
    genre_service = GenreService(database, cache)
    film_service = FilmService(database, cache)
    distribution_service = DistributionService(database, cache, cinema_service, film_service)

    data = {
        'cinema': cinema_service.get(1),
        'genres': genre_service.get_all()
    }
    data['screens'] = screen_service.get_all(cinema_id=1)

    games = 100
    output = f'./data/games_{games}.csv'
    write_header = not os.path.exists(output)

    agent_films = distribution_service.get_agent_films(cinema_id=1)
    film_mask = np.random.choice(list(agent_films.keys()), 20)

    times = []

    with open(output, 'a') as f:
        writer: DictWriter = None

        for i in range(1, games + 1):
            print(f'Game {i} of {games}')

            film_data = {key: agent_films[key] for key in film_mask}
            agent = Agent(data['cinema'], data['screens'], data['genres'], film_data)
            lookup = {}
            screens = {}
            films = {}

            for film_id in agent.film_data.keys():
                film, _, _ = agent.film_data[film_id]
                encoding = get_film_encoding(film, agent.cinema, agent.genres)
                films[film_id] = encoding

            for screen in agent.screens:
                encoding = get_screen_encoding(screen)
                screens[screen.id] = encoding

            agent.reset()
            while True:
                if agent.done():
                    break
                prior_state = agent.calculate_state()
                start_time = timeit.default_timer()
                session = agent.take_action()
                new_state = agent.calculate_state()
                end_time = timeit.default_timer()
                times.append(end_time - start_time)

                entry = {'label': session.x}

                screen = agent.screens[session.y]
                items, labels = screens[screen.id]
                for item, label in zip(items, labels):
                    entry[label] = item

                film = session.film
                items, labels = films[film.id]
                for item, label in zip(items, labels):
                    entry[label] = item

                x, y, z = session.x, session.y, list(
                    agent.film_data.keys()).index(session.film.id)
                for i in range(len(prior_state[session.y, :, z])):
                    entry[f'prob_{i}'] = prior_state[session.y, i, z]

                entry = {'hash': super_hash(entry)} | entry
                if (lookup.get(entry['hash']) != None):
                    continue

                if (writer is None):
                    fields = list(entry.keys())
                    writer = DictWriter(f, fieldnames=fields)

                if write_header:
                    writer.writeheader()
                    write_header = False

                writer.writerow(entry)
                lookup[entry['hash']] = 1

        f.close()

        print(f'Agent state + action iterations: {len(times)} calls, avg {np.mean(times)} seconds')
