"""Provides dependency injection containers
"""

from dependency_injector.containers import (DeclarativeContainer,
                                            WiringConfiguration)
from dependency_injector.providers import Factory, Singleton

from app.dependencies.caching import MemoryCache
from app.dependencies.data import (CinemaService, Database,
                                   DistributionService, FilmService,
                                   GenreService, ScreenService)
from app.dependencies.models import ModelService


class Container(DeclarativeContainer):
    """Main container for the API
    """

    wiring_config = WiringConfiguration(
        modules=[
            "app.routers.models",
            "app.routers.data"
        ])

    mem_cache = Singleton(
        MemoryCache
    )

    db = Singleton(
        Database
    )

    cinema_service = Factory(
        CinemaService,
        database=db,
        cache=mem_cache
    )

    screen_service = Factory(
        ScreenService,
        database=db,
        cache=mem_cache
    )

    film_service = Factory(
        FilmService,
        database=db,
        cache=mem_cache
    )

    genre_service = Factory(
        GenreService,
        database=db,
        cache=mem_cache
    )

    distribution_service = Factory(
        DistributionService,
        database=db,
        cache=mem_cache,
        cinema_service=cinema_service,
        film_service=film_service
    )

    model_service = Singleton(
        ModelService
    )
