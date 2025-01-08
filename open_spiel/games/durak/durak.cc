// Copyright 2025
// C++ Durak implementation for OpenSpiel
//
// durak.cc: logic, state transitions, observer, etc.

#include "open_spiel/games/durak.h"

#include <algorithm>
#include <random>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace durak {
namespace {

// GameType registration. 
const GameType kGameType{
    /*short_name=*/"durak",
    /*long_name=*/"Durak",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{},
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/true
};

// Corresponding GameInfo for usage in c++.
const GameInfo kGameInfo{
    /*num_distinct_actions=*/kNumCards + 3,  // 36 card-play actions + 3 extras
    /*max_chance_outcomes=*/kNumCards,
    /*num_players=*/kNumPlayers,
    /*min_utility=*/-1.0,
    /*max_utility=*/1.0,
    /*utility_sum=*/0.0,
    /*max_game_length=*/300
};

// Factory for creating the game from parameters.
std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<DurakGame>(params);
}

// We register the game with OpenSpiel's internal catalog.
REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// -----------------------------------------------------------------------------
// DurakGame implementation
// -----------------------------------------------------------------------------

DurakGame::DurakGame(const GameParameters& params)
    : Game(kGameType, kGameInfo, params) {
  // No special parameters in this example. Could parse them if needed.
}

std::unique_ptr<State> DurakGame::NewInitialState() const {
  return std::unique_ptr<State>(new DurakState(shared_from_this()));
}

std::vector<int> DurakGame::InformationStateTensorShape() const {
  // A simplistic shape example: 
  // See the Python for references. We have up to 1-hot for trump_suit (4),
  // 1-hot for RoundPhase (4), 1-hot for trump_card (36), deck_size, player bits, etc.
  // plus optionally the entire hand (36). 
  // The final shape is up to your design. For example:
  //     [2] (which player) + 4 (trump_suit) + 4 (phase) + 1 (deck_size) +
  //     2 (attacker_ind, defender_ind) + 36 (trump_card) + 36 (my_cards) + ...
  // We'll just define a possible dimension that matches the Python approach.
  // If you want a more compact or different shape, you can adjust accordingly.

  // As a rough total:
  //   player: 2
  //   trump_suit: 4
  //   phase: 4
  //   deck_size: 1
  //   attacker_ind + defender_ind: 2
  //   trump_card: 36
  //   my_cards: 36
  //   table_attack + table_defense: 36 + 36 = 72  (if perfect recall or public info)
  // You can refine or split it. We'll just unify with the python logic, which had
  // a total "flat" dimension of 2 + 4 + 4 + 1 + 1 + 1 + 36 + 36 + 36 + 36 = 157 or so
  // depending on which info you include. 

  // For simplicity, let's choose a single large dimension. In practice, you'd be consistent
  // with how you handle iig_obs_type. If "public info" is off, we might exclude table info.
  // We'll just give a max shape for the union of all possibilities.
  return {157};  // A single vector dimension that can store all bits.
}

std::vector<int> DurakGame::ObservationTensorShape() const {
  // Typically, the same or similar shape. Possibly fewer bits if we exclude private info.
  // Let's reuse the same dimension for simplicity.
  return {157};
}

std::shared_ptr<Observer> DurakGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  // If params is non-empty, you could parse them. Here we skip that.
  return std::make_shared<DurakObserver>(
      iig_obs_type.value_or(IIGObservationType(
          /*public_info=*/false, /*perfect_recall=*/false,
          /*private_info=*/PrivateInfoType::kSinglePlayer)));
}

// -----------------------------------------------------------------------------
// DurakState implementation
// -----------------------------------------------------------------------------

DurakState::DurakState(std::shared_ptr<const Game> game)
    : State(game) {
  // Create a 36-card deck
  deck_.reserve(kNumCards);
  for (int c = 0; c < kNumCards; ++c) deck_.push_back(c);

  // Shuffle it. 
  // (OpenSpiel has a built-in function if needed, or you can do std::shuffle.)
  std::shuffle(deck_.begin(), deck_.end(), std::mt19937(std::random_device()()));

  // Initialize hands, table, discard, etc.
  for (int p = 0; p < kNumPlayers; ++p) {
    hands_[p].clear();
  }
  table_cards_.clear();
  discard_.clear();
  trump_suit_ = -1;
  trump_card_ = -1;
  cards_dealt_ = 0;
  deck_pos_ = 0;
  attacker_ = 0;
  defender_ = 1;
  phase_ = RoundPhase::kChance;
  round_starter_ = 0;
  game_over_ = false;
}

Player DurakState::CurrentPlayer() const {
  if (game_over_) return kTerminalPlayerId;
  if (phase_ == RoundPhase::kChance) return kChancePlayerId;
  // Attack or Additional => attacker, Defense => defender
  if (phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional) {
    return attacker_;
  }
  return defender_;
}

bool DurakState::IsTerminal() const {
  return game_over_;
}

std::vector<double> DurakState::Returns() const {
  // See Python logic for final scoring.
  if (!game_over_) {
    return {0.0, 0.0};
  }

  // Count how many players still hold cards.
  std::vector<int> players_with_cards;
  for (int p = 0; p < kNumPlayers; ++p) {
    if (!hands_[p].empty()) {
      players_with_cards.push_back(p);
    }
  }

  // If exactly one player still has cards => that player is the loser: -1
  // The other is winner => +1
  if (players_with_cards.size() == 1) {
    std::vector<double> result(kNumPlayers, 0.0);
    int loser = players_with_cards[0];
    result[loser] = -1.0;
    result[1 - loser] = 1.0;
    return result;
  }

  // If both have cards, or 0 with cards but deck not empty => treat as [0,0].
  if (players_with_cards.size() == 2) {
    return {0.0, 0.0};
  }

  // If neither has cards => check deck. If deck is empty => last attacker wins. 
  if (players_with_cards.empty()) {
    if (deck_pos_ >= static_cast<int>(deck_.size())) {
      // Attacker is winner
      std::vector<double> result(kNumPlayers, 0.0);
      result[attacker_] = 1.0;
      result[1 - attacker_] = -1.0;
      return result;
    } else {
      // Deck not empty => fallback or partial: return [0, 0].
      return {0.0, 0.0};
    }
  }

  // If some other scenario arises, fallback to [0,0].
  return {0.0, 0.0};
}

void DurakState::CheckGameOver() {
  // This method is called after each move to see if game is done.

  // If a player is out of cards and deck is also empty => game over
  bool p0_empty = hands_[0].empty();
  bool p1_empty = hands_[1].empty();

  if ((p0_empty || p1_empty) && deck_pos_ >= static_cast<int>(deck_.size())) {
    game_over_ = true;
    return;
  }

  // If both players have no cards
  if (p0_empty && p1_empty) {
    // If deck is also empty => game over
    if (deck_pos_ >= static_cast<int>(deck_.size())) {
      game_over_ = true;
      return;
    } else {
      // Refill
      RefillHands();
    }
  }
}

std::string DurakState::ToString() const {
  // A text representation for debugging/logging.
  std::string str;
  absl::StrAppend(&str, "Phase=", static_cast<int>(phase_),
                  " Attack=", attacker_, " Defend=", defender_,
                  " DeckPos=", deck_pos_, "/", deck_.size(),
                  " TrumpSuit=", trump_suit_,
                  " TrumpCard=", trump_card_ < 0 ? "None" : CardToString(trump_card_),
                  " game_over=", (game_over_ ? "true" : "false"), "\n");
  for (int p = 0; p < kNumPlayers; ++p) {
    absl::StrAppend(&str, "Player ", p, " hand: ");
    for (int c : hands_[p]) {
      absl::StrAppend(&str, CardToString(c), " ");
    }
    absl::StrAppend(&str, "\n");
  }
  absl::StrAppend(&str, "Table: ");
  for (auto &pair : table_cards_) {
    int ac = pair.first;
    int dc = pair.second;
    absl::StrAppend(&str, CardToString(ac), "->",
                    (dc < 0 ? "?" : CardToString(dc)), "  ");
  }
  absl::StrAppend(&str, "\nDiscard: ", discard_.size(), " cards\n");
  return str;
}

std::unique_ptr<State> DurakState::Clone() const {
  return std::unique_ptr<State>(new DurakState(*this));
}

bool DurakState::IsChanceNode() const {
  return (phase_ == RoundPhase::kChance);
}

std::vector<std::pair<Action, double>> DurakState::ChanceOutcomes() const {
  // During the initial dealing: 
  //   each card is chosen from the deck top in a single known order for the next deal.
  // Probability is 1.0 for that next "forced" card.
  // Then eventually we reveal the bottom card as trump with probability 1.0.
  // Implementation is similar to Python.

  std::vector<std::pair<Action, double>> outcomes;
  if (cards_dealt_ < kCardsPerPlayer * kNumPlayers) {
    // The next top card
    int next_card = deck_[deck_pos_];
    outcomes.push_back({next_card, 1.0});
  } else {
    // reveal the bottom card as trump
    if (trump_card_ < 0) {
      int bottom_card = deck_.back();
      outcomes.push_back({bottom_card, 1.0});
    }
  }
  return outcomes;
}

void DurakState::ApplyChanceAction(Action outcome) {
  if (cards_dealt_ < kCardsPerPlayer * kNumPlayers) {
    // Deal this card to the next player
    int player_idx = cards_dealt_ % kNumPlayers;
    hands_[player_idx].push_back(outcome);
    ++deck_pos_;
    ++cards_dealt_;
  } else {
    // Reveal the bottom as trump
    trump_card_ = outcome;
    trump_suit_ = SuitOf(outcome);
    DecideFirstAttacker();
    phase_ = RoundPhase::kAttack;
    round_starter_ = attacker_;
  }
}

void DurakState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    ApplyChanceAction(move);
    CheckGameOver();
    return;
  }

  if (game_over_) return;  // No-op if somehow action after terminal.

  Player player = CurrentPlayer();
  // Extra actions?
  if (move >= kNumCards) {
    if (move == kExtraActionTakeCards) {
      DefenderTakesCards();
    } else if (move == kExtraActionFinishAttack) {
      AttackerFinishesAttack();
    } else if (move == kExtraActionFinishDefense) {
      DefenderFinishesDefense();
    }
    CheckGameOver();
    return;
  }

  // Otherwise, it's a card index in [0..35].
  // Must be in player's hand to be valid, or we do nothing if it's invalid
  auto &hand = hands_[player];
  auto it = std::find(hand.begin(), hand.end(), move);
  if (it == hand.end()) {
    // invalid action
    return;  // We do no-op if it's not in the player's hand
  }

  // If Attack phase or Additional phase
  if ((phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional) &&
      (player == attacker_)) {
    // Place this card face up on the table (uncovered).
    hand.erase(it);
    table_cards_.push_back(std::make_pair(move, -1));
    // We remain in Attack. 
    phase_ = RoundPhase::kAttack;
  }
  // If Defense phase
  else if (phase_ == RoundPhase::kDefense && (player == defender_)) {
    // Find the earliest uncovered card
    int uncovered_index = -1;
    for (int i = 0; i < static_cast<int>(table_cards_.size()); ++i) {
      if (table_cards_[i].second < 0) {  // not covered
        uncovered_index = i;
        break;
      }
    }
    if (uncovered_index >= 0) {
      int att_card = table_cards_[uncovered_index].first;
      if (CanDefendCard(move, att_card)) {
        // Cover the card
        hand.erase(it);
        table_cards_[uncovered_index].second = move;
        // If all covered => Additional
        bool all_covered = true;
        for (auto &pair : table_cards_) {
          if (pair.second < 0) {
            all_covered = false;
            break;
          }
        }
        if (all_covered) {
          phase_ = RoundPhase::kAdditional;
        }
      }
    }
  }

  CheckGameOver();
}

std::vector<Action> DurakState::LegalActions() const {
  if (game_over_) return {};
  if (IsChanceNode()) {
    // The set of chance actions is enumerated in ChanceOutcomes().
    std::vector<Action> actions;
    auto co = ChanceOutcomes();
    for (auto &o : co) {
      actions.push_back(o.first);
    }
    return actions;
  }

  // Otherwise, normal player moves
  std::vector<Action> moves;
  Player player = CurrentPlayer();
  const auto &hand = hands_[player];

  if (phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional) {
    if (player == attacker_) {
      // The attacker can place certain cards or finish the attack.
      // Let's be consistent with the python logic about ranks to place:
      // If table is empty, any card is legal. If not empty, attacker can only
      // add ranks that appear on the table (att or def side).
      if (table_cards_.empty()) {
        // attacker can place any card
        for (int c : hand) {
          moves.push_back(c);
        }
      } else {
        // gather ranks from the table
        std::vector<int> ranks_on_table;
        ranks_on_table.reserve(table_cards_.size() * 2);
        for (auto &pair : table_cards_) {
          ranks_on_table.push_back(RankOf(pair.first));
          if (pair.second >= 0) {
            ranks_on_table.push_back(RankOf(pair.second));
          }
        }
        // attacker can only play cards whose rank is in ranks_on_table
        for (int c : hand) {
          int r = RankOf(c);
          if (std::find(ranks_on_table.begin(), ranks_on_table.end(), r)
              != ranks_on_table.end()) {
            moves.push_back(c);
          }
        }
      }
      // Also attacker can FINISH_ATTACK if at least one card is on table
      if (!table_cards_.empty()) {
        moves.push_back(kExtraActionFinishAttack);
      }
    }
  } else if (phase_ == RoundPhase::kDefense) {
    if (player == defender_) {
      // Defender can TAKE_CARDS, or try to defend each uncovered card,
      // or FINISH_DEFENSE if no uncovered remain
      // 1) check if there are uncovered
      bool any_uncovered = false;
      int earliest_uncovered = -1;
      for (int i = 0; i < static_cast<int>(table_cards_.size()); ++i) {
        if (table_cards_[i].second < 0) {
          any_uncovered = true;
          if (earliest_uncovered < 0) earliest_uncovered = i;
          // break; // If you want to cover any uncovered in any order, you'd code differently
        }
      }

      if (!any_uncovered) {
        // can FINISH_DEFENSE
        moves.push_back(kExtraActionFinishDefense);
      } else {
        // can TAKE_CARDS
        moves.push_back(kExtraActionTakeCards);
        // or attempt to cover earliest uncovered
        if (earliest_uncovered >= 0) {
          int att_card = table_cards_[earliest_uncovered].first;
          for (int c : hand) {
            if (CanDefendCard(c, att_card)) {
              moves.push_back(c);
            }
          }
        }
      }
    }
  }

  // Sorted order is nice, but not strictly required
  std::sort(moves.begin(), moves.end());
  return moves;
}

std::string DurakState::ActionToString(Player /*player*/, Action action_id) const {
  if (action_id == kExtraActionTakeCards) {
    return "TAKE_CARDS";
  } else if (action_id == kExtraActionFinishAttack) {
    return "FINISH_ATTACK";
  } else if (action_id == kExtraActionFinishDefense) {
    return "FINISH_DEFENSE";
  } else if (action_id >= 0 && action_id < kNumCards) {
    // kNumCards is 36 in your Durak
    return absl::StrCat("Play:", CardToString(action_id));
  } else {
    return absl::StrCat("UnknownAction(", action_id, ")");
  }
}

std::string DurakState::ActionToString(Player /*player*/, Action action_id) const {
  if (action_id == kExtraActionTakeCards) return "TAKE_CARDS";
  if (action_id == kExtraActionFinishAttack) return "FINISH_ATTACK";
  if (action_id == kExtraActionFinishDefense) return "FINISH_DEFENSE";
  if (action_id >= 0 && action_id < kNumCards) {
    return absl::StrCat("Play:", CardToString(action_id));
  }
  return "UnknownAction";
}

void DurakState::UndoAction(Player /*player*/, Action /*move*/) {
  // If you need to implement Undo for algorithms that rely on it, you would
  // replicate the logic from DoApplyAction in reverse. For now, we can leave it
  // unimplemented or throw an error if needed.
  SpielFatalError("UndoAction is not implemented for Durak.");
}

// Helper: decide first attacker by lowest trump card
void DurakState::DecideFirstAttacker() {
  int lowest_trump = -1;
  Player who = 0;
  for (int p = 0; p < kNumPlayers; ++p) {
    for (int c : hands_[p]) {
      if (SuitOf(c) == trump_suit_) {
        if (lowest_trump < 0 || RankOf(c) < RankOf(lowest_trump)) {
          lowest_trump = c;
          who = p;
        }
      }
    }
  }
  attacker_ = who;
  defender_ = 1 - who;
}

// Helper: can defense_card cover attack_card?
bool DurakState::CanDefendCard(int defense_card, int attack_card) const {
  int att_s = SuitOf(attack_card);
  int att_r = RankOf(attack_card);
  int def_s = SuitOf(defense_card);
  int def_r = RankOf(defense_card);

  // same suit, higher rank
  if (att_s == def_s && def_r > att_r) {
    return true;
  }
  // defend with trump if attack is not trump
  if (def_s == trump_suit_ && att_s != trump_suit_) {
    return true;
  }
  // if both trump, rank must be higher
  if (att_s == trump_suit_ && def_s == trump_suit_ && def_r > att_r) {
    return true;
  }
  return false;
}

// The defender picks up all table cards
void DurakState::DefenderTakesCards() {
  for (auto &pair : table_cards_) {
    hands_[defender_].push_back(pair.first);
    if (pair.second >= 0) {
      hands_[defender_].push_back(pair.second);
    }
  }
  table_cards_.clear();
  phase_ = RoundPhase::kAttack;
  RefillHands();
}

// The attacker decides not to lay more cards
void DurakState::AttackerFinishesAttack() {
  if (table_cards_.empty()) {
    // If no cards on table, not much to do.
    return;
  }
  phase_ = RoundPhase::kDefense;
}

// The defender says "done" if all are covered; else effectively picks up
void DurakState::DefenderFinishesDefense() {
  bool uncovered = false;
  for (auto &pair : table_cards_) {
    if (pair.second < 0) {  // not covered
      uncovered = true;
      break;
    }
  }
  if (uncovered) {
    // Takes cards
    DefenderTakesCards();
  } else {
    // move them to discard, roles swap
    for (auto &pair : table_cards_) {
      discard_.push_back(pair.first);
      if (pair.second >= 0) {
        discard_.push_back(pair.second);
      }
    }
    table_cards_.clear();
    // swap roles
    Player old_attacker = attacker_;
    attacker_ = defender_;
    defender_ = old_attacker;
    RefillHands();
    phase_ = RoundPhase::kAttack;
  }
}

// Refill each player's hand up to 6, starting with the attacker
void DurakState::RefillHands() {
  std::array<Player, kNumPlayers> order = {attacker_, defender_};
  while (deck_pos_ < static_cast<int>(deck_.size())) {
    bool all_full = true;
    for (auto p : order) {
      if (static_cast<int>(hands_[p].size()) < kCardsPerPlayer &&
          deck_pos_ < static_cast<int>(deck_.size())) {
        hands_[p].push_back(deck_[deck_pos_]);
        deck_pos_++;
      }
    }
    for (auto p : order) {
      if (static_cast<int>(hands_[p].size()) < kCardsPerPlayer) {
        all_full = false;
      }
    }
    if (all_full) break;
  }
}

// -----------------------------------------------------------------------------
// Observations: we provide minimal placeholders. 
// You can expand them to match your Python approach exactly.
// -----------------------------------------------------------------------------

std::string DurakState::InformationStateString(Player player) const {
  // For single-player private info style, it's typically the same as
  // ObservationString if we are only exposing that player's hand.
  return ObservationString(player);
}

std::string DurakState::ObservationString(Player player) const {
  // A simple textual summary from that player's viewpoint.
  std::string str = absl::StrCat("Player ", player, " viewpoint. Phase=",
                                 static_cast<int>(phase_),
                                 " Attacker=", attacker_,
                                 " Defender=", defender_, "\n");
  absl::StrAppend(&str, "Trump card: ",
                  (trump_card_ < 0 ? "None" : CardToString(trump_card_)), "\n");

  // Show my hand (private info)
  absl::StrAppend(&str, "My Hand: ");
  for (int c : hands_[player]) {
    absl::StrAppend(&str, CardToString(c), " ");
  }
  absl::StrAppend(&str, "\n");

  // Public table info
  absl::StrAppend(&str, "Table: ");
  for (auto &pair : table_cards_) {
    int ac = pair.first;
    int dc = pair.second;
    absl::StrAppend(&str, CardToString(ac), "->",
                    (dc < 0 ? "?" : CardToString(dc)), "  ");
  }
  absl::StrAppend(&str, "\n");
  absl::StrAppend(&str, "DeckRemaining=", (deck_.size() - deck_pos_), "\n");
  return str;
}

void DurakState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  // We'll do the same as ObservationTensor for now. 
  // If you want to hide the opponent's hand, do not fill that part, etc.
  ObservationTensor(player, values);
}

void DurakState::ObservationTensor(Player player, absl::Span<float> values) const {
  // Flatten into the 157-dim vector we claimed in InformationStateTensorShape.
  // For demonstration, we will simply fill with 0/1. 
  SPIEL_CHECK_EQ(values.size(), 157);
  for (int i = 0; i < 157; i++) {
    values[i] = 0.f;
  }
  // (You would replicate the Python logic that sets bits for player ID, trump_suit,
  // my_cards, table_attack, table_defense, etc.)
  // This is left as an exercise/placeholder.
}

// -----------------------------------------------------------------------------
// DurakObserver
// -----------------------------------------------------------------------------

DurakObserver::DurakObserver(IIGObservationType iig_obs_type)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      iig_obs_type_(iig_obs_type) {}

void DurakObserver::WriteTensor(const State& observed_state, int player,
                                Allocator* allocator) const {
  // The simplest approach is to replicate DurakState::ObservationTensor logic.
  // We'll just pass it through.
  // In a more refined approach, you would interpret iig_obs_type_ carefully.
  const DurakState& state = open_spiel::down_cast<const DurakState&>(observed_state);
  auto out = allocator->Get("observation", {157});
  std::vector<float> tmp(157, 0.f);
  state.ObservationTensor(player, absl::MakeSpan(tmp));
  for (int i = 0; i < 157; ++i) {
    out.at(i) = tmp[i];
  }
}

std::string DurakObserver::StringFrom(const State& observed_state,
                                      int player) const {
  // Similarly, we can just pass through to the state's method.
  const DurakState& state = open_spiel::down_cast<const DurakState&>(observed_state);
  // If iig_obs_type_ has private_info == kSinglePlayer, we show that player's hand.
  // If private_info == kNone, we might hide it, etc. 
  // For demonstration, we just do what the state does.
  return state.ObservationString(player);
}

}  // namespace durak
}  // namespace open_spiel
