"""Shared FastAPI dependencies. Pull services out of app.state so routes can `Depends` on them."""

from fastapi import Request

from app.inference import DataStore, ModelRegistry, Predictor


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_data_store(request: Request) -> DataStore:
    return request.app.state.data_store


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor
