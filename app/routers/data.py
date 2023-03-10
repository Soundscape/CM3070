"""API routes for data
"""

import random

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies.containers import Container
from app.dependencies.data import (CinemaService, DistributionService,
                                   FilmService, ScreenService)

router = APIRouter()


@router.get("/api/v1/data/cinema", tags=["Data"])
@inject
async def get_cinemas(
    cinema_service: CinemaService = Depends(Provide[Container.cinema_service])
):
    result = cinema_service.get_all()
    return result


@router.get("/api/v1/data/cinema/{cinema_id}", tags=["Data"])
@inject
async def get_cinema(
    cinema_id: int,
    cinema_service: CinemaService = Depends(Provide[Container.cinema_service])
):
    result = cinema_service.get(cinema_id)
    if result is None:
        raise HTTPException(status_code=404, detail='Cinema not found')

    return result


@router.get("/api/v1/data/film", tags=["Data"])
@inject
async def get_films(
    film_service: FilmService = Depends(Provide[Container.film_service])
):
    result = film_service.get_all()
    result = random.sample(result, k=20)
    result.sort(key=lambda d: d.title)

    return result


@router.get("/api/v1/data/film/{film_id}", tags=["Data"])
@inject
async def get_film(
    film_id: int,
    film_service: FilmService = Depends(Provide[Container.film_service])
):
    result = film_service.get(film_id)

    return result


@router.get("/api/v1/data/cinema/{cinema_id}/screen", tags=["Data"])
@inject
async def get_screens(
    cinema_id: int,
    screen_service: ScreenService = Depends(Provide[Container.screen_service])
):
    result = screen_service.get_all(cinema_id)

    return result


@router.get("/api/v1/data/film/{film_id}/distribution", tags=["Data"])
@inject
async def slot_reward(
    film_id: int,
    film_service: FilmService = Depends(Provide[Container.film_service]),
    distribution_service: DistributionService = Depends(
        Provide[Container.distribution_service])
):
    film = film_service.get(film_id)
    dist = distribution_service.get_film_distribution(film)

    return {'distribution':  dist.tolist()}
