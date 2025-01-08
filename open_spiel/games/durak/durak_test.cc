/*
 * durak_test.cc
 *
 * C++ tests for Durak in OpenSpiel.
 */

#include "open_spiel/games/durak/durak.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace durak {
namespace {

namespace testing = open_spiel::testing;

void BasicDurakTests() {
  // 1) LoadGameTest checks that we can load and clone the game, etc.
  testing::LoadGameTest("durak");

  // 2) Check chance outcomes are valid for the initial dealing & trump reveal.
  testing::ChanceOutcomesTest(*LoadGame("durak"));

  // 3) RandomSimTest does random rollouts of entire games to ensure we never
  //    crash or produce invalid states.
  testing::RandomSimTest(*LoadGame("durak"), /*num_sims=*/50);

  // If you implement Undo, test it here:
  // testing::RandomSimTestWithUndo(*LoadGame("durak"), /*num_sims=*/10);

  // For demonstration, we also create an observer:
  auto observer = LoadGame("durak")->MakeObserver(
      /*iig_obs_type=*/IIGObservationType(/*public_info=*/true,
                                          /*perfect_recall=*/false,
                                          /*private_info=*/
                                          PrivateInfoType::kSinglePlayer),
      /*params=*/{});
  testing::RandomSimTestCustomObserver(*LoadGame("durak"), observer, 10);
}

void SerializeDeserializeTest() {
  std::shared_ptr<const Game> game = LoadGame("durak");
  std::unique_ptr<State> state = game->NewInitialState();

  // Simulate some actions
  // For testing purposes, let's apply some chance actions to reach a certain state
  while (!state->IsTerminal() && state->CurrentPlayer() == kChancePlayerId) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    SPIEL_CHECK_EQ(outcomes.size(), 1);  // deterministic chance outcomes
    state->ApplyAction(outcomes[0].first);
  }

  // Apply a player action if possible
  if (!state->IsTerminal()) {
    std::vector<Action> legal_actions = state->LegalActions();
    if (!legal_actions.empty()) {
      state->ApplyAction(legal_actions[0]);
    }
  }

  // Test serialization and deserialization
  testing::TestSerializeDeserialize(*game, state.get());
}

void CountStatesTest() {
  std::shared_ptr<const Game> game = LoadGame("durak");

  // Example to get all states without a depth limit:
  auto states = algorithms::GetAllStates(
      *game,
      /*depth_limit=*/-1,
      /*include_terminals=*/true,
      /*include_chance_states=*/true);
  // For Durak, the state space can be large. Just ensure it completes.
  std::cout << "Number of reachable states: " << states.size() << std::endl;
}

// main test driver
}  // namespace
}  // namespace durak
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::durak::BasicDurakTests();
  open_spiel::durak::SerializeDeserializeTest();
  open_spiel::durak::CountStatesTest();
  return 0;
}
