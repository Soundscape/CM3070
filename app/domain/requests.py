from pydantic import BaseModel, Field


class SessionInput(BaseModel):
    """
    Summary of session placement
    """
    slot: int = Field(..., example=32, title='starting slot index')
    film: int = Field(..., example=1, title='film identifier')
    screen: int = Field(..., example=1, title='screen identifier')


class PredictRequest(BaseModel):
    """
    Input values for prediction
    """
    screens: list[int] = Field(..., example=[6, 2, 1],
                               title='screen identifiers')
    films: list[int] = Field(..., example=[1, 4, 6, 8],
                             title='film identifiers')
    slots: tuple[int, int] = Field(..., example=[32, 88], title='slot range')
    sessions: list[SessionInput] = Field(..., example=[SessionInput(slot=38, film=1, screen=1), SessionInput(
        slot=72, film=4, screen=1)], title='previously placed sessions')
