// Copyright 2025
// C++ Durak-with-transfers implementation for OpenSpiel
//
// durak_with_transfers.cc: logic, state transitions, observer, etc.

#include "open_spiel/games/durak/durak_with_transfers.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace durak_with_transfers {
namespace {

// -----------------------------------------------------------------------------
// Register a new GameType to differentiate from the original "durak".
// -----------------------------------------------------------------------------

const GameType kGameType{
    /*short_name=*/"durak_with_transfers",
    /*long_name=*/"Durak with transfers",
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
    /*parameter_specification=*/{
      {"init_deck", GameParameter(std::string(""))},
      {"rng_seed", GameParameter(0)},
    },
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/true
};

const GameInfo kGameInfo{
    /*num_distinct_actions=*/kNumCards + 4,  // up from +3 to +4 because we added TRANSFER
    /*max_chance_outcomes=*/kNumCards,
    /*num_players=*/kNumPlayers,
    /*min_utility=*/-1.0,
    /*max_utility=*/1.0,
    /*utility_sum=*/0.0,
    /*max_game_length=*/300
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<DurakWithTransfersGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// -----------------------------------------------------------------------------
// Shuffling helper (unchanged).
// -----------------------------------------------------------------------------
void ShuffleDeck(std::mt19937* rng, std::vector<int>& deck, int begin, int end) {
  for (int i = begin; i < end - 1; ++i) {
    int j = i + (*rng)() % (end - i);
    std::swap(deck[i], deck[j]);
  }
}

// -----------------------------------------------------------------------------
// DurakWithTransfersGame
// -----------------------------------------------------------------------------

DurakWithTransfersGame::DurakWithTransfersGame(const GameParameters& params)
    : Game(kGameType, params),
      rng_seed_(ParameterValue<int>("rng_seed")) {}

std::unique_ptr<State> DurakWithTransfersGame::NewInitialState() const {
  return std::unique_ptr<State>(
      new DurakWithTransfersState(shared_from_this(), rng_seed_));
}

std::vector<int> DurakWithTransfersGame::InformationStateTensorShape() const {
  // We reuse the dimension from the original approach: 157 or so.
  return {157};
}

std::vector<int> DurakWithTransfersGame::ObservationTensorShape() const {
  return {157};
}

std::shared_ptr<Observer> DurakWithTransfersGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  IIGObservationType obs_type = iig_obs_type.value_or(IIGObservationType());
  obs_type.public_info = false;
  obs_type.perfect_recall = false;
  obs_type.private_info = PrivateInfoType::kSinglePlayer;

  return std::make_shared<DurakWithTransfersObserver>(obs_type);
}

// -----------------------------------------------------------------------------
// DurakWithTransfersState
// -----------------------------------------------------------------------------

DurakWithTransfersState::DurakWithTransfersState(std::shared_ptr<const Game> game,
                                                 int rng_seed)
    : State(game), rng_seed_(rng_seed) {
  const DurakWithTransfersGame* g =
      down_cast<const DurakWithTransfersGame*>(game.get());
  auto param_map = g->GetParameters();
  auto it = param_map.find("init_deck");
  std::string deck_str =
      (it != param_map.end()) ? it->second.string_value() : "";

  if (deck_str.empty()) {
    // Initialize a standard ordered deck
    for (int i = 0; i < kNumCards; i++) {
      deck_.push_back(i);
    }

    // Create a reproducible random engine
    std::mt19937 rng(rng_seed_);
    ShuffleDeck(&rng, deck_, 0, kNumCards);
  } else {
    // Parse the deck string
    std::stringstream ss(deck_str);
    for (int i = 0; i < kNumCards; i++) {
      int c;
      char comma;
      ss >> c;
      deck_.push_back(c);
      if (i < kNumCards - 1) ss >> comma;  // consume comma
    }
  }

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

Player DurakWithTransfersState::CurrentPlayer() const {
  if (game_over_) return kTerminalPlayerId;
  if (phase_ == RoundPhase::kChance) return kChancePlayerId;
  // Attack or Additional => attacker, Defense => defender
  if (phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional) {
    return attacker_;
  }
  return defender_;
}

bool DurakWithTransfersState::IsTerminal() const {
  return game_over_;
}

std::vector<double> DurakWithTransfersState::Returns() const {
  if (!game_over_) {
    return {0.0, 0.0};
  }

  // Same logic as original Durak: if exactly one player has cards => loser/winner
  // if both/none => other checks.
  std::vector<int> players_with_cards;
  for (int p = 0; p < kNumPlayers; ++p) {
    if (!hands_[p].empty()) {
      players_with_cards.push_back(p);
    }
  }
  if (players_with_cards.size() == 1) {
    std::vector<double> result(kNumPlayers, 0.0);
    int loser = players_with_cards[0];
    result[loser] = -1.0;
    result[1 - loser] = 1.0;
    return result;
  }
  if (players_with_cards.size() == 2) {
    return {0.0, 0.0};
  }
  if (players_with_cards.empty()) {
    // if deck is empty => attacker wins, else 0.0
    if (deck_pos_ >= static_cast<int>(deck_.size())) {
      std::vector<double> result(kNumPlayers, 0.0);
      result[attacker_] = 1.0;
      result[1 - attacker_] = -1.0;
      return result;
    } else {
      return {0.0, 0.0};
    }
  }

  return {0.0, 0.0};
}

std::string DurakWithTransfersState::ToString() const {
  std::string str;
  absl::StrAppend(&str, "Phase=", static_cast<int>(phase_),
                  " Attack=", attacker_, " Defend=", defender_,
                  " DeckPos=", deck_pos_, "/", deck_.size(),
                  " TrumpSuit=", trump_suit_,
                  " TrumpCard=",
                  (trump_card_ < 0 ? "None" : CardToString(trump_card_)),
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

std::unique_ptr<State> DurakWithTransfersState::Clone() const {
  return std::unique_ptr<State>(new DurakWithTransfersState(*this));
}

void DurakWithTransfersState::UndoAction(Player /*player*/, Action /*move*/) {
  SpielFatalError("UndoAction is not implemented for Durak-with-transfers.");
}

bool DurakWithTransfersState::IsChanceNode() const {
  return (phase_ == RoundPhase::kChance);
}

std::vector<std::pair<Action, double>> DurakWithTransfersState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  if (cards_dealt_ < kCardsPerPlayer * kNumPlayers) {
    int next_card = deck_[deck_pos_];
    outcomes.push_back({next_card, 1.0});
  } else {
    if (trump_card_ < 0) {
      int bottom_card = deck_.back();
      outcomes.push_back({bottom_card, 1.0});
    }
  }
  return outcomes;
}

void DurakWithTransfersState::ApplyChanceAction(Action outcome) {
  if (cards_dealt_ < kCardsPerPlayer * kNumPlayers) {
    int player_idx = cards_dealt_ % kNumPlayers;
    hands_[player_idx].push_back(outcome);
    ++deck_pos_;
    ++cards_dealt_;
  } else {
    trump_card_ = deck_.back();
    trump_suit_ = SuitOf(deck_.back());
    DecideFirstAttacker();
    phase_ = RoundPhase::kAttack;
    round_starter_ = attacker_;
  }
}

void DurakWithTransfersState::DoApplyAction(Action move) {
  last_action_ = move;  // track the last action

  if (IsChanceNode()) {
    ApplyChanceAction(move);
    CheckGameOver();
    return;
  }
  if (game_over_) return;

  // extra action: TRANSFER
  if (move == kExtraActionTransfer) {
    DefenderTransfers();
    CheckGameOver();
    return;
  }

  // extra actions: TAKE_CARDS, FINISH_ATTACK, FINISH_DEFENSE
  if (move == kExtraActionTakeCards) {
    DefenderTakesCards();
    CheckGameOver();
    return;
  }
  if (move == kExtraActionFinishAttack) {
    AttackerFinishesAttack();
    CheckGameOver();
    return;
  }
  if (move == kExtraActionFinishDefense) {
    DefenderFinishesDefense();
    CheckGameOver();
    return;
  }

  // Otherwise, it's a card ID (0..35)
  Player player = CurrentPlayer();
  auto &hand = hands_[player];
  auto it = std::find(hand.begin(), hand.end(), move);
  if (it == hand.end()) {
    return;  // invalid
  }

  // Attacker playing a card
  if ((phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional)
      && player == attacker_) {
    hand.erase(it);
    table_cards_.push_back(std::make_pair(move, -1));
    phase_ = RoundPhase::kAttack;  // remain in Attack phase
  }
  // Defender covering a card
  else if (phase_ == RoundPhase::kDefense && player == defender_) {
    int uncovered_index = -1;
    for (int i = 0; i < static_cast<int>(table_cards_.size()); ++i) {
      if (table_cards_[i].second < 0) {
        uncovered_index = i;
        break;
      }
    }
    if (uncovered_index >= 0) {
      int att_card = table_cards_[uncovered_index].first;
      if (CanDefendCard(move, att_card)) {
        hand.erase(it);
        table_cards_[uncovered_index].second = move;
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

std::vector<Action> DurakWithTransfersState::LegalActions() const {
  if (game_over_) return {};
  if (IsChanceNode()) {
    // Return the forced dealing outcome(s)
    std::vector<Action> actions;
    auto co = ChanceOutcomes();
    for (auto &o : co) {
      actions.push_back(o.first);
    }
    return actions;
  }

  std::vector<Action> moves;
  Player player = CurrentPlayer();
  const auto &hand = hands_[player];

  // Attacker's actions
  if ((phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional)
      && player == attacker_) {
    // 1) The attacker can place a new attacking card (with rank restriction if table has cards)
    if (table_cards_.empty()) {
      // can place any card
      for (int c : hand) {
        moves.push_back(c);
      }
    } else {
      // can only place ranks that appear on the table, up to kCardsPerPlayer or fewer if the defender has fewer
      if (static_cast<int>(table_cards_.size()) < kCardsPerPlayer && !hands_[defender_].empty()) {
        std::vector<int> ranks_on_table;
        ranks_on_table.reserve(table_cards_.size() * 2);
        for (auto &pair : table_cards_) {
          ranks_on_table.push_back(RankOf(pair.first));
          if (pair.second >= 0) {
            ranks_on_table.push_back(RankOf(pair.second));
          }
        }
        for (int c : hand) {
          int r = RankOf(c);
          if (std::find(ranks_on_table.begin(), ranks_on_table.end(), r)
              != ranks_on_table.end()) {
            moves.push_back(c);
          }
        }
      }
    }

    // 2) The attacker can FINISH_ATTACK if there's at least one card on the table
    //    *and* we haven't just done a TRANSFER last move
    if (!table_cards_.empty() && last_action_ != kExtraActionTransfer) {
      moves.push_back(kExtraActionFinishAttack);
    }
  }
  // Defender's actions in RoundPhase::kDefense
  else if (phase_ == RoundPhase::kDefense && player == defender_) {
    bool any_uncovered = false;
    bool any_covered = false;
    for (auto &pair : table_cards_) {
      if (pair.second < 0) {
        any_uncovered = true;
      } else {
        any_covered = true;
      }
    }
    if (!any_uncovered) {
      // everything is covered => FINISH_DEFENSE
      moves.push_back(kExtraActionFinishDefense);
    } else {
      // can TAKE_CARDS
      moves.push_back(kExtraActionTakeCards);

      // can TRANSFER if no card is covered yet (all are uncovered)
      if (any_uncovered && !any_covered) {
        // see if the defender has a rank matching the attacked card(s)
        std::unordered_set<int> ranks_on_table;
        for (auto &pair : table_cards_) {
          ranks_on_table.insert(RankOf(pair.first));
        }
        for (int c : hand) {
          if (ranks_on_table.find(RankOf(c)) != ranks_on_table.end()) {
            moves.push_back(kExtraActionTransfer);
            break;
          }
        }
      }

      // or try to cover the earliest uncovered
      int earliest_uncovered_idx = -1;
      for (int i = 0; i < static_cast<int>(table_cards_.size()); ++i) {
        if (table_cards_[i].second < 0) {
          earliest_uncovered_idx = i;
          break;
        }
      }
      if (earliest_uncovered_idx >= 0) {
        int att_card = table_cards_[earliest_uncovered_idx].first;
        for (int c : hand) {
          if (CanDefendCard(c, att_card)) {
            moves.push_back(c);
          }
        }
      }
    }
  }

  std::sort(moves.begin(), moves.end());
  return moves;
}

std::string DurakWithTransfersState::ActionToString(Player /*player*/, Action action_id) const {
  if (action_id == kExtraActionTakeCards) return "TAKE_CARDS";
  if (action_id == kExtraActionFinishAttack) return "FINISH_ATTACK";
  if (action_id == kExtraActionFinishDefense) return "FINISH_DEFENSE";
  if (action_id == kExtraActionTransfer)     return "TRANSFER";
  if (action_id >= 0 && action_id < kNumCards) {
    return absl::StrCat("Play:", CardToString(action_id));
  }
  return "UnknownAction";
}

// -----------------------------------------------------------------------------
// Helper methods
// -----------------------------------------------------------------------------

void DurakWithTransfersState::DecideFirstAttacker() {
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

bool DurakWithTransfersState::CanDefendCard(int defense_card, int attack_card) const {
  int att_s = SuitOf(attack_card);
  int att_r = RankOf(attack_card);
  int def_s = SuitOf(defense_card);
  int def_r = RankOf(defense_card);

  if (att_s == def_s && def_r > att_r) {
    return true;
  }
  if (def_s == trump_suit_ && att_s != trump_suit_) {
    return true;
  }
  if (att_s == trump_suit_ && def_s == trump_suit_ && def_r > att_r) {
    return true;
  }
  return false;
}

void DurakWithTransfersState::DefenderTakesCards() {
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

void DurakWithTransfersState::AttackerFinishesAttack() {
  if (table_cards_.empty()) {
    return;
  }
  phase_ = RoundPhase::kDefense;
}

void DurakWithTransfersState::DefenderFinishesDefense() {
  bool uncovered = false;
  for (auto &pair : table_cards_) {
    if (pair.second < 0) {
      uncovered = true;
      break;
    }
  }
  if (uncovered) {
    DefenderTakesCards();
  } else {
    for (auto &pair : table_cards_) {
      discard_.push_back(pair.first);
      if (pair.second >= 0) {
        discard_.push_back(pair.second);
      }
    }
    table_cards_.clear();
    Player old_attacker = attacker_;
    attacker_ = defender_;
    defender_ = old_attacker;
    RefillHands();
    phase_ = RoundPhase::kAttack;
  }
}

void DurakWithTransfersState::RefillHands() {
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

void DurakWithTransfersState::CheckGameOver() {
  bool p0_empty = hands_[0].empty();
  bool p1_empty = hands_[1].empty();

  if ((p0_empty || p1_empty) && deck_pos_ >= static_cast<int>(deck_.size())) {
    game_over_ = true;
    return;
  }
  if (p0_empty && p1_empty) {
    if (deck_pos_ >= static_cast<int>(deck_.size())) {
      game_over_ = true;
      return;
    } else {
      RefillHands();
    }
  }
}

// The defender "transfers" by swapping roles, going to kAdditional
void DurakWithTransfersState::DefenderTransfers() {
  // We assume we've checked in LegalActions() that it's valid:
  Player old_attacker = attacker_;
  attacker_ = defender_;
  defender_ = old_attacker;
  phase_ = RoundPhase::kAdditional;
}

// Observations (same placeholders as original)
std::string DurakWithTransfersState::InformationStateString(Player player) const {
  return ObservationString(player);
}

std::string DurakWithTransfersState::ObservationString(Player player) const {
  std::string str = absl::StrCat("Player ", player, " viewpoint. Phase=",
                                 static_cast<int>(phase_),
                                 " Attacker=", attacker_,
                                 " Defender=", defender_, "\n");
  absl::StrAppend(&str, "Trump card: ",
                  (trump_card_ < 0 ? "None" : CardToString(trump_card_)), "\n");
  absl::StrAppend(&str, "My Hand: ");
  for (int c : hands_[player]) {
    absl::StrAppend(&str, CardToString(c), " ");
  }
  absl::StrAppend(&str, "\nTable: ");
  for (auto &pair : table_cards_) {
    int ac = pair.first;
    int dc = pair.second;
    absl::StrAppend(&str, CardToString(ac), "->",
                    (dc < 0 ? "?" : CardToString(dc)), "  ");
  }
  absl::StrAppend(&str, "\nDeckRemaining=", (deck_.size() - deck_pos_), "\n");
  return str;
}

void DurakWithTransfersState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  ObservationTensor(player, values);
}

void DurakWithTransfersState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), 157);
  for (int i = 0; i < 157; i++) {
    values[i] = 0.f;
  }
}

// -----------------------------------------------------------------------------
// DurakWithTransfersObserver
// -----------------------------------------------------------------------------

DurakWithTransfersObserver::DurakWithTransfersObserver(IIGObservationType iig_obs_type)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      iig_obs_type_(iig_obs_type) {}

void DurakWithTransfersObserver::WriteTensor(const State& observed_state,
                                             int player,
                                             Allocator* allocator) const {
  const DurakWithTransfersState& state =
      open_spiel::down_cast<const DurakWithTransfersState&>(observed_state);
  auto out = allocator->Get("observation", {157});
  std::vector<float> tmp(157, 0.f);
  state.ObservationTensor(player, absl::MakeSpan(tmp));
  for (int i = 0; i < 157; ++i) {
    out.at(i) = tmp[i];
  }
}

std::string DurakWithTransfersObserver::StringFrom(
    const State& observed_state, int player) const {
  const DurakWithTransfersState& state =
      open_spiel::down_cast<const DurakWithTransfersState&>(observed_state);
  return state.ObservationString(player);
}

}  // namespace durak_with_transfers
}  // namespace open_spiel
