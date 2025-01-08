/*
 * durak_test.cc
 *
 * C++ tests for Durak in OpenSpiel.
 */

#include "open_spiel/games/durak/durak.h"

// Standard OpenSpiel includes:
#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"  // For RandomSimTest, etc.

namespace open_spiel {
namespace durak {
namespace {

// A helper to test that serialization & deserialization yield an equivalent
// game and state. This is what 'TestSerializeDeserialize' used to do in some
// branches of OpenSpiel.
void MySerializeDeserializeTest(const Game& game, const State& state) {
  // Serialize the game & current state
  std::string serialized = SerializeGameAndState(game, state);

  // Deserialize to a new (game, state) pair
  auto game_and_state = DeserializeGameAndState(serialized);

  // Compare the string representations to ensure equality
  SPIEL_CHECK_EQ(game.ToString(), game_and_state.first->ToString());
  SPIEL_CHECK_EQ(state.ToString(), game_and_state.second->ToString());
}

// ----------------------------------------------------------------------------
// Basic tests
// ----------------------------------------------------------------------------

void BasicDurakTests() {
  // 1) LoadGameTest checks that we can load and clone the game, etc.
  testing::LoadGameTest("durak");

  // 2) Check chance outcomes are valid for the initial dealing & trump reveal.
  testing::ChanceOutcomesTest(*LoadGame("durak"));

  // 3) RandomSimTest does random rollouts of entire games to ensure we never
  //    crash or produce invalid states.
  testing::RandomSimTest(*LoadGame("durak"), /*num_sims=*/50);

  // For demonstration, we create a custom observer with some IIG settings:
  auto game = LoadGame("durak");

  IIGObservationType iig_obs_type;
  iig_obs_type.public_info = true;
  iig_obs_type.perfect_recall = false;
  iig_obs_type.private_info = PrivateInfoType::kSinglePlayer;

  auto observer = game->MakeObserver(iig_obs_type, /*params=*/{});
  testing::RandomSimTestCustomObserver(*game, observer);
}

// ----------------------------------------------------------------------------
// Test serialization / deserialization
// ----------------------------------------------------------------------------

void SerializeDeserializeTest() {
  std::shared_ptr<const Game> game = LoadGame("durak");
  std::unique_ptr<State> state = game->NewInitialState();

  // Let's advance the state through any forced chance actions (dealing)
  while (!state->IsTerminal() && state->CurrentPlayer() == kChancePlayerId) {
    std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
    SPIEL_CHECK_EQ(outcomes.size(), 1);  // we expect exactly 1 forced outcome
    state->ApplyAction(outcomes[0].first);
  }

  // Then let the current player (if not terminal) take one action.
  if (!state->IsTerminal()) {
    std::vector<Action> legal_actions = state->LegalActions();
    if (!legal_actions.empty()) {
      state->ApplyAction(legal_actions[0]);  // pick the first, arbitrary
    }
  }

  // Now we do our custom check for serialization & deserialization:
  MySerializeDeserializeTest(*game, *state);
}

// ----------------------------------------------------------------------------
// (Optional) CountStatesTest: enumerates all reachable states up to infinite depth.
// ----------------------------------------------------------------------------

void CountStatesTest() {
  std::shared_ptr<const Game> game = LoadGame("durak");

  // Example to get all states without a depth limit:
  auto states = algorithms::GetAllStates(
      *game,
      /*depth_limit=*/-1,
      /*include_terminals=*/true,
      /*include_chance_states=*/true);

  // For Durak, the state space can be large. We just show it completes:
  std::cout << "Number of reachable states: " << states.size() << std::endl;
}

// ----------------------------------------------------------------------------
// Main test driver
// ----------------------------------------------------------------------------

}  // namespace
}  // namespace durak
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::durak::BasicDurakTests();
  open_spiel::durak::SerializeDeserializeTest();
  // Runs for too long, commented out for now
  // open_spiel::durak::CountStatesTest();
  return 0;
}
