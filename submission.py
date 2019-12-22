from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    if not state.snakes[player_index].alive or len(state.fruits_locations) == 0:
        return state.snakes[player_index].length

    head = state.snakes[player_index].head
    tail = state.snakes[player_index].tail_position
    borders = [(state.board_size.height, head[1]), (head[0], state.board_size.width),
               (head[0], 0), (0, head[1])]
    remaining_turns = state.game_duration_in_turns - state.turn_number + 1  # +1 to prevent divide by zero
    dist = lambda snake, fruit: ((snake[0] - fruit[0])**2 + (snake[1] - fruit[1])**2)**0.5
    dist_to_closest_fruit = min([dist(head, s) for s in state.fruits_locations])
    dist_to_closest_border = min([dist(head, border) for border in borders])
    num_of_possible_fruits = len([dist(head, s) < remaining_turns for s in state.fruits_locations])
    board_area = state.board_size.height * state.board_size.width
    dist_to_closest_opp = min([dist(tail, s.head) - dist(head, s.head) for s in state.snakes if s.index != player_index]
                              + [board_area])
    stay_straight = sum([dist(head, s) for s in state.snakes[player_index].position])
    return state.snakes[player_index].length + \
           2*(1 - dist_to_closest_fruit/board_area) + \
           (1 - remaining_turns/state.game_duration_in_turns)*(num_of_possible_fruits / len(state.fruits_locations)) + \
           (1 - remaining_turns/state.game_duration_in_turns)*(dist_to_closest_opp / board_area) + \
           max((state.snakes[player_index].length/remaining_turns - 0.2), 0)*stay_straight + \
           0.2*dist_to_closest_border/board_area


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """
        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        pass


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        pass


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()

