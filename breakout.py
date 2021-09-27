from enum import Enum

class GameState(Enum):
    GAME_ACTIVE = 1
    GAME_MENU = 2
    GAME_WIN = 3


class Game:
    def __init__(self, width, height):
        self.game_state = GameState.GAME_ACTIVE

    def process_input(self):
        pass

    def render(self):
        pass


class Shader:
    def compile(self, vertex_source, fragment_source, geometry_source=None):
        pass

    def use(self):
        pass

class Texture:
    pass