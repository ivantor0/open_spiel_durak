# durak_test.py
#
# Basic Python tests for Durak game in OpenSpiel.

from absl.testing import absltest
import pyspiel
from open_spiel.python.observation import make_observation
from open_spiel.python.algorithms.get_all_states import get_all_states
import numpy as np


class DurakGameTest(absltest.TestCase):

  def test_load_and_random_playouts(self):
    """Loads the game and does random playouts."""
    game = pyspiel.load_game("python_durak")
    # Basic sanity check: do some random rollouts.
    pyspiel.random_sim_test(game, num_sims=5, serialize=False, verbose=True)

  def test_consistent(self):
    """Checks the Python and C++ Durak implementations are the same."""

    python_game = pyspiel.load_game("python_durak")
    cc_game = pyspiel.load_game("durak")

    # We'll compare with two observation types, e.g. perfect_recall or default.
    obs_types = [
        None,
        pyspiel.IIGObservationType(perfect_recall=True)
    ]

    # Construct observers
    py_observers = [make_observation(python_game, o) for o in obs_types]
    cc_observers = [make_observation(cc_game, o) for o in obs_types]

    # Gather all states for each implementation
    python_states = get_all_states(
        python_game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=True
    )
    cc_states = get_all_states(
        cc_game,
        depth_limit=-1,
        include_terminals=True,
        include_chance_states=True
    )

    # Check that the sets of states (by string key) match
    self.assertCountEqual(list(python_states), list(cc_states))

    # For each key (history of actions), verify we get the same returns,
    # and the same observation tensors for each player.
    for key, cc_state in cc_states.items():
      py_state = python_states[key]
      # compare returns
      np.testing.assert_array_equal(py_state.returns(), cc_state.returns())
      # compare history
      np.testing.assert_array_equal(py_state.history(), cc_state.history())

      # compare observations for each observer type + each player
      for py_obs, cc_obs in zip(py_observers, cc_observers):
        for player_id in range(py_state.num_players()):
          py_obs.set_from(py_state, player_id)
          cc_obs.set_from(cc_state, player_id)
          np.testing.assert_array_equal(py_obs.tensor, cc_obs.tensor)

  def test_observation_and_chance(self):
    """Verifies that chance nodes and observation objects function properly."""
    game = pyspiel.load_game("python_durak")

    # Create a couple of different observation configurations:
    obs_type1 = pyspiel.IIGObservationType(
        public_info=True, perfect_recall=False,
        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER)
    obs_type2 = None  # default observer

    obs1 = make_observation(game, obs_type1)
    obs2 = make_observation(game, obs_type2)

    state = game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        # pick the only chance outcome (prob=1) from ChanceOutcomes
        outcomes = state.chance_outcomes()
        action_list = [a for (a, _) in outcomes]
        # This is usually forced (prob=1 each step of the initial dealing),
        # so we just pick the first:
        state.apply_action(action_list[0])
      else:
        # For each player, set the observer and check we can access the data.
        current_player = state.current_player()
        obs1.set_from(state, current_player)
        obs2.set_from(state, current_player)
        # We won't do an elaborate check, just ensure the shapes are correct:
        self.assertEqual(len(obs1.tensor), len(obs2.tensor))

        # pick a random legal move:
        actions = state.legal_actions()
        self.assertTrue(actions)  # should never be empty until terminal
        state.apply_action(actions[0])

    # If we reach here, we never crashed or did something inconsistent.

  def test_reach_some_states(self):
    """Ensures we can call get_all_states. For large games, might be big, so limit depth."""
    game = pyspiel.load_game("python_durak")
    states = get_all_states(
        game,
        depth_limit=20,  # limit so we don't blow up
        include_terminals=True,
        include_chance_states=True)
    # Just a basic check that we got some states:
    self.assertGreater(len(states), 0)

  # Optionally add more tests checking final returns, etc.


if __name__ == "__main__":
  absltest.main()
