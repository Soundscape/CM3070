from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """
    Output of prediction
    """
    slot: int = Field(..., example=32, title='starting slot index')
    screen: int = Field(..., example=1, title='screen index')
    runtime: int = Field(..., example=16, title='film runtime')
    reward: float = Field(..., example=1.237, title='action reward')
    film: int = Field(..., example=1, title='film identifier')


class PredictBulkResponse(BaseModel):
    """
    Output of prediction
    """
    sessions: list[PredictResponse] = Field(...,
                                            title='The suggested sessions')
