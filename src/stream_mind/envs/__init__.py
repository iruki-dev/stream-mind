from .base import GameModel, RealtimeEnv
from .freeway import Freeway
from .snake import Snake

GAMES = {"freeway": Freeway, "snake": Snake}

__all__ = ["GameModel", "RealtimeEnv", "Freeway", "Snake", "GAMES"]
