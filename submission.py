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
    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length

    dist = lambda snake, fruit: ((snake[0] - fruit[0]) ** 2 + (snake[1] - fruit[1]) ** 2) ** 0.5
    danger = lambda positions, fruit: ((fruit[0]-1, fruit[1]) in positions) + ((fruit[0]+1, fruit[1]) in positions) + ((fruit[0], fruit[1]-1) in positions) + ((fruit[0], fruit[1]+1) in positions)
    length = state.snakes[player_index].length
    head = state.snakes[player_index].head
    tail = state.snakes[player_index].tail_position
    width = state.board_size.width
    height = state.board_size.height
    area = state.board_size.height * state.board_size.width
    close_borders = [(height-1, head[1]), (head[0], width-1), (head[0], 0), (0, head[1])]
    remaining_turns = state.game_duration_in_turns - state.turn_number + 1  # +1 to prevent divide by zero
    total_turns = state.game_duration_in_turns
    total_fruits = len(state.fruits_locations) + 1

    dist_to_closest_fruit = min([dist(head, s) for s in state.fruits_locations + [(height**2, height**2)] if danger(state.snakes[player_index].position, s) < 3])  # range:[1, sqrt(2)*height]
    # num_of_possible_fruits = len([dist(head, s) < remaining_turns for s in state.fruits_locations])  # range:[0, fruits]
    dist_to_closest_opp = min([(dist(head, s.tail_position) + dist(head, s.head))/2 for s in state.snakes if s.index != player_index] + [area])  # range:[1, sqrt(2)*height]
    stay_straight = min([dist(head, s) for s in state.snakes[player_index].position[1:3]])  # range:[1, length/2]
    dist_to_closest_border = min([dist(head, border) for border in close_borders])  # range:[0, height/2]

    return length + \
           (1 - dist_to_closest_fruit/(height*2**0.5)) + \
           0.7*(1 - remaining_turns/total_turns)*(dist_to_closest_opp / (height*2**0.5)) + \
           0.6*max(((length / total_fruits) - 3), 0.1) * (stay_straight / (length / 2)) + \
           0.5*max(((length/total_fruits) - 3), 0.01)*(dist_to_closest_border/(height/2))


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

    def MinMax_calc(self, state: TurnBasedGameState, d: int) -> float:
    # Leaf level or the depth is over - return heuristic (len(gameState.getLegalActions()) == 0? no need in run_game)
    #if (not state.terminal(player_index=player_index)) or (d == 0):
        if d == 0:
            return heuristic(state.game_state, self.player_index)
        # our agent turn
        if state.turn == self.Turn.AGENT_TURN:
            d = d - 1
            currMax = -np.inf
            for action in state.game_state.get_possible_actions(player_index=self.player_index):
                #next_state = get_next_state(state.game_state, state.game_state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index)[self.player_index])
                # next_state = get_next_state(state.game_state, action)
                next_state_TurnBasedGameState = self.TurnBasedGameState(state.game_state, action)
                # will be sent again to the MinMax_calc with our chosen action (the next turn is for OPPONENTS)
                h_value = self.MinMax_calc(next_state_TurnBasedGameState, d)
                # h_value = self._heuristic(next_state)
                # if found a better score than currMax- update current max and max action that matches the current max
                if h_value > currMax:
                    currMax = h_value
            return currMax

        else:
            # the OPPONENTS turn
            currMin = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action  # ?
                next_state = get_next_state(state.game_state, opponents_actions) # נשלח שוב למינימקס עד לעומק הרצוי, וחוזר לשלנו ב-NONE
                next_state_TurnBasedGameState = self.TurnBasedGameState(next_state, None)
                # will be sent again to the MinMax_calc with None (the next turn is for our agent)
                h_value = self.MinMax_calc(next_state_TurnBasedGameState, d)
                # h_value = self._heuristic(next_state)
                # if found a lower score than currMax- update current max and max action that matches the current max
                if h_value < currMin:
                    currMin = h_value
            return currMin

    def get_action(self, state: GameState) -> GameAction:
        d = 3

        state_leftAction = self.TurnBasedGameState(state, GameAction.LEFT)
        state_rightAction = self.TurnBasedGameState(state, GameAction.RIGHT)
        state_straightAction = self.TurnBasedGameState(state, GameAction.STRAIGHT)

        value_leftAction = self.MinMax_calc(state_leftAction, d)
        value_rightAction= self.MinMax_calc(state_rightAction, d)
        value_straightAction = self.MinMax_calc(state_straightAction, d)

        if value_leftAction > value_rightAction:
            if value_leftAction > value_straightAction:
                return state_leftAction.agent_action
        else:
            if value_rightAction > value_straightAction:
                return state_rightAction.agent_action
        return state_straightAction.agent_action

    def _heuristic(self, state: GameState) -> float:
        return heuristic(state, self.player_index)



class AlphaBetaAgent(MinimaxAgent):

    def AlphaBeta_calc(self, state: MinimaxAgent.TurnBasedGameState, d: int, Alpha: int, Beta: int) -> float:
        # Leaf level or the depth is over - return heuristic (len(gameState.getLegalActions()) == 0? no need in run_game)
        if d == 0:
            return heuristic(state.game_state, self.player_index)
        # our agent turn
        if state.turn == self.Turn.AGENT_TURN:
            d = d - 1
            currMax = -np.inf
            for action in state.game_state.get_possible_actions(player_index=self.player_index):
                #next_state = get_next_state(state.game_state, state.game_state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index)[self.player_index])
                # next_state = get_next_state(state.game_state, action)
                next_state_TurnBasedGameState = self.TurnBasedGameState(state.game_state, action)
                # will be sent again to the AlphaBeta_calc with our chosen action (the next turn is for OPPONENTS)
                h_value = self.AlphaBeta_calc(next_state_TurnBasedGameState, d, Alpha, Beta)
                # h_value = self._heuristic(next_state)
                # if found a better score than currMax- update current max and max action that matches the current max
                if h_value > currMax:
                    currMax = h_value
                Alpha = max(currMax, Alpha)
                if currMax >= Beta:
                    # pruning
                    return np.inf
            return currMax
        else:
            # the OPPONENTS turn
            currMin = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action  # ?
                next_state = get_next_state(state.game_state, opponents_actions)
                # will be sent again to the AlphaBeta_calc with None (the next turn is for our agent)
                next_state_TurnBasedGameState = self.TurnBasedGameState(next_state, None)
                h_value = self.AlphaBeta_calc(next_state_TurnBasedGameState, d, Alpha, Beta)
                # h_value = self._heuristic(next_state)
                # if found a lower score than currMax- update current max and max action that matches the current max
                if h_value < currMin:
                    currMin = h_value
                Beta = min(currMin, Beta)
                if currMin <= Alpha:
                    # pruning
                    return -np.inf
            return currMin


    def get_action(self, state: GameState) -> GameAction:
        d = 3
        Alpha = -np.inf
        Beta = np.inf

        state_leftAction = self.TurnBasedGameState(state, GameAction.LEFT)
        state_rightAction = self.TurnBasedGameState(state, GameAction.RIGHT)
        state_straightAction = self.TurnBasedGameState(state, GameAction.STRAIGHT)

        value_leftAction = self.AlphaBeta_calc(state_leftAction, d, Alpha, Beta)
        value_rightAction= self.AlphaBeta_calc(state_rightAction, d, Alpha, Beta)
        value_straightAction = self.AlphaBeta_calc(state_straightAction, d, Alpha, Beta)

        if value_leftAction > value_rightAction:
            if value_leftAction > value_straightAction:
                return state_leftAction.agent_action
        else:
            if value_rightAction > value_straightAction:
                return state_rightAction.agent_action
        return state_straightAction.agent_action


    def _heuristic(self, state: GameState) -> float:
        return heuristic(state, self.player_index)

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
    n = 50
    actions = np.asarray(list(GameAction))
    current = np.random.choice(actions, size=n)
    sideways = 0
    for i in range(1, n):
        best_val = -np.inf
        best_states = []
        for action in actions:
            if current[i] == action:
                continue
            new = current
            new[i] = action
            new_val = get_fitness(tuple(new))
            if new_val > best_val:
                best_val = new_val
                best_states = [new]
            elif new_val == best_val:
                best_states.append(new)
        if best_val > get_fitness(tuple(current)):
            random_index = np.random.choice(len(best_states))
            current = best_states[random_index]
            sideways = 0
        elif best_val == get_fitness(tuple(current)) and sideways < np.inf:
            random_index = np.random.choice(len(best_states))
            current = best_states[random_index]
            sideways += 1
        else:
            print("Found max before exhaustion")
            print("Current fitness = " + str(get_fitness(tuple(current))))
            print("Current move vector = " + str(current))
            return
    print("Found max")
    print("Current fitness = " + str(get_fitness(tuple(current))))
    print("Current move vector = " + str(current))


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

