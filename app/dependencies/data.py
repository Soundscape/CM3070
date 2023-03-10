"""Module which provides the data access layer"""

import numpy as np
from prisma import Prisma
from prisma.models import Cinema, Film, Genre, GenreGaussian, Screen

from app.dependencies.caching import MemoryCache
from models.utils import generate_gaussian, get_film_slots


class Database:
    """An object which handles connectivity to a Prisma defined database
    """

    def __init__(self) -> None:
        self.db = Prisma()

    def __del__(self) -> None:
        if self.db.is_connected():
            self.db.disconnect()

    def get_db(self) -> Prisma:
        """Retrieve the database handle"""
        if not self.db.is_connected():
            self.db.connect()
        return self.db


class CinemaService:
    """Cinema repository
    """

    def __init__(
        self,
        database: Database,
        cache: MemoryCache
    ) -> None:
        self.database = database
        self.cache = cache

    def get_all(self) -> list[Cinema]:
        """Retrieve all cinemas

        Returns:
            list[Cinema]: The list of cinemas
        """
        db = self.database.get_db()
        key = 'cinemas'
        result: Cinema = self.cache.get(key, Cinema.parse_obj)
        if result is None:
            result = db.cinema.find_many()
            self.cache.set(key, result)

        return result

    def get(self, cinema_id: int) -> Cinema | None:
        """Retrieve a cinema by its ID

        Args:
            cinema_id (int): The cinema ID

        Returns:
            Cinema | None: The cinema
        """
        db = self.database.get_db()
        key = f'cinema_{cinema_id}'
        result: Cinema = self.cache.get(key, Cinema.parse_obj)
        if result is None:
            result = db.cinema.find_unique(where={'id': cinema_id})
            if result is None:
                return None
            self.cache.set(key, result)

        return result


class ScreenService:
    """Screen repository
    """

    def __init__(
        self,
        database: Database,
        cache: MemoryCache
    ) -> None:
        self.database = database
        self.cache = cache

    def get_all(self, cinema_id: int) -> list[Screen]:
        """Retrieves all screens at a cinema

        Args:
            cinema_id (int): The cinema ID

        Returns:
            list[Screen]: The cinema's screens
        """
        db = self.database.get_db()
        key = f'cinema_{cinema_id}_screens'
        result: Screen = self.cache.get(key, Screen.parse_obj)
        if result is None:
            result = db.screen.find_many(
                where={'cinemaId': cinema_id}, order={'capacity': 'desc'})
            self.cache.set(key, result)

        return result

    def get(self, screen_id: int) -> Screen | None:
        """Retrieve a screen by its ID

        Args:
            screen_id (int): The screen ID

        Returns:
            Screen | None: The screen
        """
        db = self.database.get_db()
        key = f'screen_{screen_id}'
        result: Screen = self.cache.get(key, Screen.parse_obj)
        if result is None:
            result = db.screen.find_unique(where={'id': screen_id})
            if result is None:
                return None
            self.cache.set(key, result)

        return result


class FilmService:
    """Film repository
    """

    def __init__(
        self,
        database: Database,
        cache: MemoryCache
    ) -> None:
        self.database = database
        self.cache = cache

    def get_all(self) -> list[Film]:
        """Retrieve all films

        Returns:
            list[Film]: The films
        """
        db = self.database.get_db()
        key = 'films'
        result: Film = self.cache.get(key, Film.parse_obj)
        if result is None:
            result = db.film.find_many(
                order={'id': 'asc'}, include={'genres': True})
            self.cache.set(key, result)

        return result

    def get(self, film_id: int) -> Film | None:
        """Retrieve a film by its ID

        Args:
            film_id (int): The film ID

        Returns:
            Film | None: The film
        """
        db = self.database.get_db()
        key = f'film_{film_id}'
        result: Film = self.cache.get(key, Film.parse_obj)
        if result is None:
            result = db.film.find_unique(
                where={'id': film_id}, include={'genres': True, })
            if result is None:
                return None
            self.cache.set(key, result)

        return result


class GenreService:
    """Genre repository
    """

    def __init__(
        self,
        database: Database,
        cache: MemoryCache
    ) -> None:
        self.database = database
        self.cache = cache

    def get_all(self) -> list[Genre]:
        """Retrieves all genres

        Returns:
            list[Genre]: The genres
        """
        db = self.database.get_db()
        key = 'genres'
        result: Genre = self.cache.get(key, Genre.parse_obj)
        if result is None:
            result = db.genre.find_many(order={'name': 'asc'})
            self.cache.set(key, result)

        return result


class DistributionService:
    """Distribution repository
    """

    def __init__(
        self,
        database: Database,
        cache: MemoryCache,
        cinema_service: CinemaService,
        film_service: FilmService
    ) -> None:
        self.database = database
        self.cache = cache
        self.cinema_service = cinema_service
        self.film_service = film_service

    def get_distributions_by_genre(self, genre_id: int) -> list[GenreGaussian]:
        """Retrieve the distribution for the specified genre

        Args:
            genre_id (int): The genre ID

        Returns:
            list[GenreGaussian]: The distribution
        """
        db = self.database.get_db()
        key = f'genre_{genre_id}_distribution'
        result = self.cache.get(key, GenreGaussian.parse_obj)
        if result is None:
            result = db.genregaussian.find_many(where={'genreId': genre_id})
            self.cache.set(key, result)

        return result

    def get_film_distribution(self, film: Film) -> list[float]:
        """Retrieve the distribution for a film.
        Films may have multiple genres which are combined to form the film's distribution.

        Args:
            film (Film): The film

        Returns:
            list[float]: The distribution
        """
        distributions: list[GenreGaussian] = []
        for genre in film.genres:
            distributions += self.get_distributions_by_genre(genre.id)

        weights = np.array([d.weight for d in distributions])
        weights = weights / weights.sum()
        mu = [d.mu for d in distributions]
        sigma = [d.sigma for d in distributions]

        return generate_gaussian(mu, sigma, weights)

    def get_film_distributions(self) -> dict[int, list[GenreGaussian]]:
        """Retrieves the unprocessed distributions for all films in a single command.

        Returns:
            dict[int, list[GenreGaussian]]: The distributions
        """
        db = self.database.get_db()
        query = '''
        SELECT *
        FROM _FilmToGenre as fg
            JOIN GenreGaussian as gg ON fg.B = gg.id 
        '''
        result = {}
        query_result = db.query_raw(query)
        for item in query_result:
            film_id = str(item['A'])
            if film_id not in result:
                result[film_id] = []

            dist = GenreGaussian.parse_obj(item)
            result[film_id].append(dist)

        return result

    def get_agent_films(self, cinema_id: int) -> dict[int, tuple[Film, int, list[float]]]:
        """Retrieves the film, slots, and distributions of each film at a cinema

        Args:
            cinema_id (int): The cinema ID

        Returns:
            dict[int, tuple[Film, int, list[float]]]: The film dictionary result
        """
        result = {}

        cinema = self.cinema_service.get(cinema_id)
        films = self.film_service.get_all()
        distributions = self.get_film_distributions()

        for film in films:
            dists = distributions[str(film.id)]
            weights = np.array([d.weight for d in dists])
            weights = weights / weights.sum()
            mu = [d.mu for d in dists]
            sigma = [d.sigma for d in dists]

            slots = get_film_slots(film, cinema)
            distribution = generate_gaussian(mu, sigma, weights)
            result[film.id] = (film, slots, distribution)

        return result
