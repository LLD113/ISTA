import itertools
import numpy as np
import fuzzing.src.image_transforms as image_transforms
import copy

from fuzzing.src.utility import find_the_distance, init_image_plots, update_image_plots
from fuzzing.src.reward import Reward_Status

import pprint

pp = pprint.PrettyPrinter()



class DeepSmartFuzzer_State:
    def __init__(self, mutated_input, action=0, previous_state=None, reward_status=Reward_Status.NOT_AVAILABLE,
                 reward=None, game=None):

        self.mutated_input = copy.deepcopy(mutated_input)


        self.previous_state = previous_state
        if self.previous_state != None:
            self.original_input = copy.deepcopy(previous_state.original_input)
            self.level = previous_state.level + 1
            self.action_history = previous_state.action_history + [action]
            self.game = previous_state.game
        else:
            self.original_input = copy.deepcopy(mutated_input)
            self.level = 0
            self.action_history = []
            self.game = game

        self.nb_actions = self.game.get_nb_actions(self.level)
        self.game_finished = False
        self.reward_status = reward_status
        if self.reward_status != Reward_Status.NOT_AVAILABLE:
            self.input_changed = True
        else:
            self.input_changed = False
        self.reward = reward

    def visit(self):
        if self.reward_status == Reward_Status.UNVISITED:
            self.reward_status = Reward_Status.VISITED



class DeepSmartFuzzer:
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.coverage = experiment.coverage
        self.input_shape = params.input_shape
        self.input_lower_limit = params.input_lower_limit
        self.input_upper_limit = params.input_upper_limit
        options_p1 = []
        self.actions_p1_spacing = []
        for i in range(len(params.action_division_p1)):
            spacing = int(self.input_shape[i] / params.action_division_p1[i])
            if self.input_shape[i] == 1:
                options_p1.append([0])
            else:
                options_p1.append(list(range(0, self.input_shape[i] - spacing + 1, spacing)))
            self.actions_p1_spacing.append(spacing)

        actions_p1_lower_limit = np.array(list(itertools.product(*options_p1)))
        actions_p1_upper_limit = np.add(actions_p1_lower_limit, self.actions_p1_spacing)


        for i in range(len(params.action_division_p1)):
            if self.input_shape[i] != 1:
                round_up = actions_p1_upper_limit[:, i] > (self.input_shape[i] - self.actions_p1_spacing[i])
                actions_p1_upper_limit[:, i] = round_up * self.input_shape[i] + np.logical_not(
                    round_up) * actions_p1_upper_limit[:, i]

        self.actions_p1 = []
        for i in range(len(actions_p1_lower_limit)):
            self.actions_p1.append({
                "lower_limits": actions_p1_lower_limit[i],
                "upper_limits": actions_p1_upper_limit[i]
            })

        self.actions_p2 = params.actions_p2
        self.ending_condition = params.tc3
        self.with_implicit_reward = params.implicit_reward
        self.verbose = params.verbose
        self.image_verbose = params.image_verbose

        self.best_reward = 0
        self.best_input = None

        if self.verbose:
            print("self.actions_p1")
            pp.pprint(self.actions_p1)
            print("self.actions_p2")
            pp.pprint(self.actions_p2)

        if self.image_verbose:
            self.f_current = init_image_plots(8, 8, self.input_shape)
            self.f_best = init_image_plots(8, 8, self.input_shape)


    def get_stat(self):
        return self.best_reward, self.best_input


    def reset_stat(self):
        self.best_reward = 0
        self.best_input = None


    def player(self, level):
        if level % 2 == 0:
            return 1
        else:
            return 2


    def get_nb_actions(self, level):
        if self.player(level) == 1:
            return len(self.actions_p1)
        else:
            return len(self.actions_p2)


    def apply_action(self, state, action1, action2):
        mutated_input = copy.deepcopy(state.mutated_input)


        action_part1 = self.actions_p1[action1]
        action_part2 = self.actions_p2[action2]


        lower_limits = action_part1['lower_limits']
        upper_limits = action_part1['upper_limits']
        s = tuple([slice(lower_limits[i], upper_limits[i]) for i in range(len(lower_limits))])


        for j in range(len(mutated_input)):
            mutated_input_piece = np.zeros(147392)
            mutated_input_piece = mutated_input[j].reshape(self.input_shape)
            if not isinstance(action_part2, tuple):
                mutated_input_piece[s] += action_part2
            else:
                f = getattr(image_transforms, 'image_' + action_part2[0])
                m_shape = mutated_input_piece[s].shape
                i = mutated_input_piece[s].reshape(m_shape[-3:])
                i = f(i, action_part2[1])
                mutated_input_piece[s] = i.reshape(m_shape)
            mutated_input_piece[s] = np.clip(mutated_input_piece[s], self.input_lower_limit, self.input_upper_limit)

        return mutated_input


    def calc_reward(self, mutated_input):
        _, reward = self.coverage.step(mutated_input, update_state=False,
                                       with_implicit_reward=self.with_implicit_reward)


        return reward







    def step(self, state, action, return_reward=True):
        if self.player(state.level) == 1:
            new_state = DeepSmartFuzzer_State(state.mutated_input, action=action, previous_state=state,
                                              reward_status=Reward_Status.NOT_AVAILABLE)
        else:
            action1 = state.action_history[-1]
            action2 = action
            mutated_input = self.apply_action(state, action1, action2)
            reward = self.calc_reward(mutated_input)
            new_state = DeepSmartFuzzer_State(mutated_input, action=action, previous_state=state,
                                              reward_status=Reward_Status.UNVISITED, reward=reward)

        if self.ending_condition(new_state):

            new_state.game_finished = True

        if not new_state.game_finished and new_state.reward != None:
            if self.verbose:
                print("Reward:", new_state.reward)

            if self.image_verbose:
                title = "level:" + str(new_state.level) + " Action: " + str((action1, action2)) + " Reward: " + str(
                    new_state.reward)
                update_image_plots(self.f_current, new_state.mutated_input, title)

            if new_state.reward > self.best_reward:
                self.best_input, self.best_reward = copy.deepcopy(new_state.mutated_input), new_state.reward

                if self.image_verbose:
                    title = "Best Reward: " + str(self.best_reward)
                    update_image_plots(self.f_best, self.best_input, title)

        return new_state, new_state.reward

    def print_status(self):
        print("Best Reward:", self.best_reward)
