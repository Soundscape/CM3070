"""The API entry point which defines the server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.dependencies.containers import Container
from app.routers import models, data


def create_server() -> FastAPI:
    """Initialises a FastAPI server
    """
    container = Container()

    instance = FastAPI()
    instance.container = container
    instance.add_middleware(GZipMiddleware, minimum_size=500)
    instance.add_middleware(CORSMiddleware, allow_origins=[
        '*'], allow_methods=['*'], allow_headers=['*'])
    instance.include_router(models.router)
    instance.include_router(data.router)

    return instance


server = create_server()
