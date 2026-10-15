from .base import GameModel, RealtimeEnv
from .freeway import Freeway
from .snake import Snake

from functools import partial

# 인지 부하 단계 (Real-Time Reasoning Gym처럼): 기본은 높음, -lite는 낮음.
GAMES = {
    "freeway": Freeway,
    "snake": Snake,
    "freeway-lite": partial(Freeway, width=8, max_speed=2, lanes=4, max_cars=1),
    "snake-lite": partial(Snake, size=6),
}

__all__ = ["GameModel", "RealtimeEnv", "Freeway", "Snake", "GAMES"]
