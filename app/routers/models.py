"""API routes for models
"""

import timeit

import numpy as np
import torch
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from app.dependencies.containers import Container
from app.dependencies.data import (CinemaService, DistributionService,
                                   GenreService, ScreenService)
from app.dependencies.models import ModelService
from app.domain.requests import PredictRequest
from app.domain.responses import PredictBulkResponse, PredictResponse
from models.agents import Agent
from models.classifiers import ScheduleClassifier
from models.utils import (Session, get_film_encoding, get_probabilities,
                          get_screen_encoding)

router = APIRouter()


@router.post("/api/v1/models/tune/{n_trials}/{epochs}", tags=["Model"])
@inject
async def tune_model(
    n_trials: int,
    epochs: int,
    model_service: ModelService = Depends(Provide[Container.model_service])
):
    """Perform tuning on the dataset using Optuna and ASHA

    Args:
        n_trials (int): The number of trials to run
        epochs (int): The number of epochs to train
        model_service (ModelService, optional): The model service. Defaults to Depends(Provide[Container.model_service]).

    Returns:
        _type_: The best hyper-parameters found during the trials
    """

    result = model_service.tune(n_trials, epochs)
    return result


@router.post("/api/v1/models/test/{epochs}", tags=["Model"])
@inject
async def test_model(
    epochs: int,
    model_service: ModelService = Depends(Provide[Container.model_service])
):
    """Trains the model using train, validation, and test sets. Indicates how the model performs on unseen data.

    Args:
        epochs (int): The number of epochs to train
        model_service (ModelService, optional): The model service. Defaults to Depends(Provide[Container.model_service]).

    Returns:
        dict: The training metrics
    """
    result = model_service.test(epochs)
    return result


@router.post("/api/v1/models/train", tags=["Model"])
@inject
async def train_model(
    epochs: int,
    model_service: ModelService = Depends(Provide[Container.model_service])
):
    """Trains the model on all data. Used to create the final model

    Args:
        epochs (int): The number of epochs to train
        model_service (ModelService, optional): The model service. Defaults to Depends(Provide[Container.model_service]).

    Returns:
        dict: The training metrics
    """
    result = model_service.train(epochs)
    return result


@router.post("/api/v1/models/predict/{cinema_id}", tags=["Model"], response_model=PredictBulkResponse)
@inject
async def predict_model(
    cinema_id: int,
    payload: PredictRequest,
    model_service: ModelService = Depends(Provide[Container.model_service]),
    cinema_service: CinemaService = Depends(Provide[Container.cinema_service]),
    screen_service: ScreenService = Depends(Provide[Container.screen_service]),
    genre_service: GenreService = Depends(Provide[Container.genre_service]),
    distribution_service: DistributionService = Depends(
        Provide[Container.distribution_service])
):
    """Suggests placement of session given the user-specified limits, such as time, screen, and film subsets.

    Args:
        cinema_id (int): The cinema ID
        payload (PredictRequest): The request payload
        model_service (ModelService, optional): The model service. Defaults to Depends(Provide[Container.model_service]).
        cinema_service (CinemaService, optional): The cinema service. Defaults to Depends(Provide[Container.cinema_service]).
        screen_service (ScreenService, optional): The screen service. Defaults to Depends(Provide[Container.screen_service]).
        genre_service (GenreService, optional): The genre service. Defaults to Depends(Provide[Container.genre_service]).
        distribution_service (DistributionService, optional): The distribution service. Defaults to Depends( Provide[Container.distribution_service]).

    Returns:
        PredictBulkResponse: The suggested session placements
    """
    config = model_service.get_config()
    model = ScheduleClassifier(config)
    model.load_from_checkpoint('./model.checkpoint', config=config)

    cinema = cinema_service.get(cinema_id)
    screens = screen_service.get_all(cinema_id)
    agent_screens = [d for d in screens if d.id in payload.screens]
    genres = genre_service.get_all()
    agent_films = distribution_service.get_agent_films(cinema_id)
    film_data = {key: agent_films[key] for key in payload.films}
    sessions = []
    for d in payload.sessions:
        y = next((i for i, item in enumerate(
            screens) if item.id == d.screen), -1)
        film, slots, _ = agent_films[d.film]
        session = Session(x=d.slot, y=y, runtime=slots, film=film)
        sessions.append(session)
    agent = Agent(cinema, agent_screens, genres, film_data)
    api_start = timeit.default_timer()
    agent.sessions = sessions
    agent.calculate_state()

    suggested_sessions = []

    model_times = []
    while not agent.done():
        probs = np.zeros(agent.state.shape)
        probs[:, :, -1] = agent.state[:, :, -1]

        for y, screen in enumerate(agent_screens):
            for z, film_id in enumerate(agent.film_data.keys()):
                film, slots, distribution = film_data[film_id]
                screen_encoding, _ = get_screen_encoding(screen)
                film_encoding, _ = get_film_encoding(film, cinema, genres)
                state = agent.state[y, :, z]

                x = screen_encoding + film_encoding + state.tolist()
                x = torch.tensor(x).reshape((1, len(x)))

                model_start = timeit.default_timer()
                curr_probs = model.forward(x)
                probs[y, :, z] = curr_probs.detach().numpy()
                probs[y, :, z][probs[y, :, z] == 0] = 1e-4
                mask = agent.state[y, :, z] > 0
                probs[y, :, z] = probs[y, :, z] * mask
                probs[y, :, z] = probs[y, :, z] * (1 - agent.state[y, :, -1])
                probs[y, :payload.slots[0], z] = 0
                probs[y, payload.slots[1] - (slots - 1):, z] = 0
                model_end = timeit.default_timer()
                model_times.append(model_end - model_start)

        dist = get_probabilities(probs)
        if hasattr(dist, 'sum'):
            break
        # idx = np.argmax(dist.probs)
        idx = dist.sample()
        y, x, z = np.unravel_index(idx, shape=(
            agent.state.shape[0], agent.state.shape[1], agent.state.shape[2] - 1))
        session = agent.take_determined_action(x, y, z)
        agent.calculate_state()
        suggested_sessions.append(session)

    sessions = []
    for s in suggested_sessions:
        screen = screen_service.get(payload.screens[s.y])
        film, slots, distribution = film_data[s.film.id]
        reward = screen.capacity * distribution[s.x]
        sessions.append(PredictResponse(
            slot=s.x,
            screen=payload.screens[s.y],
            runtime=s.runtime,
            film=s.film.id,
            reward=reward
        ))

    api_end = timeit.default_timer()

    print(
        f'Model state + action iterations: {len(model_times)} calls, avg {np.mean(model_times)} seconds')
    print(f'API: {api_end - api_start} seconds')

    result = PredictBulkResponse(sessions=sessions)

    return result
