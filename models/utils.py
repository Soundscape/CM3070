from math import ceil, floor

import numpy as np
import torch
import torch.nn.functional as F
from prisma.models import Cinema, Film, Genre, Screen
from pydantic import BaseModel, Field


class Session(BaseModel):
    """A film session placed in a schedule"""
    x: int = Field(..., ge=0, lt=96, title='X',
                   description='The slot number at which the session starts')
    y: int = Field(..., ge=0, title='Y',
                   description='The screen number at which the session plays')
    runtime: int = Field(..., gt=0, title='Runtime',
                         description='The runtime in slots')
    film: Film = Field(..., title='Film',
                       description='The film played during the session')


def get_screen_encoding(screen: Screen) -> tuple[list[int], list[str]]:
    """Retrieves a feature encoding of a screen"""
    labels = ['screen_capacity']
    data = [screen.capacity]

    return data, labels


def get_film_encoding(film: Film, cinema: Cinema, genres: list[Genre]) -> tuple[list[int], list[str]]:
    """Retrieves a feature encoding of a film"""
    genres, genre_labels = get_film_genre_encoding(film, genres)
    labels = ['film_slots', 'film_rating'] + [
        f'film_{d.lower()}' for d in genre_labels]
    data = [get_film_slots(film, cinema), film.rating] + genres

    return data, labels


def get_film_genre_encoding(film: Film, genres: list[Genre]) -> tuple[list[int], list[str]]:
    """Retrieves a feature encoding of a film's genres"""
    genre_names = [d.name for d in genres]
    result = [0] * len(genre_names)

    for genre in film.genres:
        idx = genre_names.index(genre.name)
        result[idx] = 1

    return result, genre_names


def gaussian(x, mu: float, sigma: float):
    """Retrieves an array based on the specified Gaussian distribution"""
    return (np.pi * sigma) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def gaussian_mixture(x, mu: list[float], sigma: list[float], weights: list[float]):
    """Mixes multiple Gaussian distributions into a single distribution"""
    return np.sum([weights[i] * gaussian(x, mu[i], sigma[i]) for i in range(len(mu))], axis=0)


def generate_gaussian(mu: list[float], sigma: list[float], weights: list[float]):
    """Generates a single Gaussian distribution from multiple Gaussian distributions"""
    x = np.linspace(0, 23, 96)
    y = gaussian_mixture(x, mu, sigma, weights)
    y = y / y.sum()

    return y


def get_film_slots(film: Film, cinema: Cinema) -> int:
    """Calculate the number of interval slots a film requires"""
    time = film.runtime + cinema.cleaningTime
    return int(ceil(time / cinema.interval))


def get_cinema_intervals(cinema: Cinema) -> tuple[int, int, int]:
    """Calculate the total, opening, and closing slots for a cinema"""
    slots = int(24 * 60 / cinema.interval)
    opens = int(cinema.opens * 60 / cinema.interval)
    closes = int(cinema.closes * 60 / cinema.interval)

    return (slots, opens, closes)


def get_spatial_map(cinema: Cinema, sessions: list[Session]):
    """Calculate a matrix indicating free and full slots based on placed sessions"""
    slots, opens, closes = get_cinema_intervals(cinema)
    spatial_map = np.zeros((slots, ))

    for session in sessions:
        start, stop = session.x, session.x + session.runtime
        assert start >= opens and stop <= closes, "Session is outside of business hours"
        spatial_map[start: stop] = 1

    return spatial_map


def get_computed_spatial_map(spatial_map, cinema: Cinema, film: Film):
    """Calculates a spatial map that accounts for boundaries around sessions and business hours"""
    _, opens, closes = get_cinema_intervals(cinema)
    runtime = get_film_slots(film, cinema)

    film_spatial_map = spatial_map.copy()

    # Set business hour restrictions
    film_spatial_map[:opens] = 1
    film_spatial_map[closes:] = 1

    # Set session boundaries
    for y in range(opens, closes):
        if all(film_spatial_map[y:y+2] == [0, 1]):
            film_spatial_map[-runtime+y+2:y+2] = 1

    # Prevent placements before closing without the required space
    film_spatial_map[-(runtime - 1):] = 1

    return film_spatial_map


def get_probability_map(scaled_distribution, computed_spatial_map):
    """Converts a film distribution and spatial map into a probability map"""
    scaled_distribution += 1e-8
    prob_map = (1 - computed_spatial_map) * scaled_distribution
    prob_map = get_probability_categorical(prob_map)

    if hasattr(prob_map, 'probs'):
        return prob_map.probs.numpy()

    return prob_map


def get_probabilities(state):
    """Converts the state into a probability distribution"""
    prob_map = state[:, :, :-1].ravel()
    prob_map = get_probability_categorical(prob_map)

    return prob_map


def get_probability_categorical(prob_map):
    """Rescales a probability map to a probability distribution"""
    if prob_map.sum() == 0:
        return prob_map

    prob_map[prob_map == 0] = -float('inf')
    prob_map = torch.tensor(prob_map)
    prob_map = F.softmax(prob_map, dim=0)
    prob_map = torch.distributions.Categorical(prob_map)

    return prob_map


def to_time(x, interval):
    """Converts a slot index to a time string"""
    hours = floor(x * interval / 60)
    minutes = int(((x * interval / 60) - hours) * 60)
    return f'{hours:02}:{minutes:02}'
