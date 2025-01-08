// Copyright 2025
// C++ Durak implementation for OpenSpiel
//
// durak.cc: logic, state transitions, observer, etc.

#include "open_spiel/games/durak/durak.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace durak {
namespace {

// Construct a GameType object with enough info for standard usage.
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
    /*parameter_specification=*/{
      {"init_deck", GameParameter(std::string(""))},
      {"rng_seed", GameParameter(0)},
    },
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/true
};

const GameInfo kGameInfo{
    /*num_distinct_actions=*/kNumCards + 3,
    /*max_chance_outcomes=*/kNumCards,
    /*num_players=*/kNumPlayers,
    /*min_utility=*/-1.0,
    /*max_utility=*/1.0,
    /*utility_sum=*/0.0,
    /*max_game_length=*/300
};

// Register the game with OpenSpiel
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
    : Game(kGameType, params),
      rng_seed_(ParameterValue<int>("rng_seed")) {
  // If "init_deck" was not specified, we fill it with a freshly shuffled deck.
  std::string init_deck = ParameterValue<std::string>("init_deck", "");
  if (init_deck.empty()) {
    std::vector<int> deck(kNumCards);
    for (int i = 0; i < kNumCards; i++) deck[i] = i;
    // Create a reproducible random engine.
    std::mt19937 rng(rng_seed_);

    ShuffleDeck(&rng, /*deck=*/deck, /*begin=*/0, /*end=*/kNumCards);

    std::ostringstream oss;
    for (int i = 0; i < kNumCards; i++) {
      if (i > 0) oss << ",";
      oss << deck[i];
    }
    init_deck = oss.str();
    game_parameters_["init_deck"] = GameParameter(init_deck);
  }
}

std::unique_ptr<State> DurakGame::NewInitialState() const {
  return std::unique_ptr<State>(new DurakState(shared_from_this(), rng_seed_));
}

void DurakGame::ShuffleDeck(std::mt19937* rng, std::vector<int> deck, int begin, int end) {
  for (int i = begin; i < end - 1; ++i) {
    int j = i + (*rng)() % (end - i);
    std::swap(deck[i], deck[j]);
  }
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

  // As a rough total:
  //   player: 2
  //   trump_suit: 4
  //   phase: 4
  //   deck_size: 1
  //   attacker_ind + defender_ind: 2
  //   trump_card: 36
  //   my_cards: 36
  //   table_attack + table_defense: 36 + 36 = 72  (if perfect recall or public info)

  // For simplicity, let's choose a single large dimension.
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
  // Construct or fill out the fields on IIGObservationType:
  IIGObservationType obs_type = iig_obs_type.value_or(IIGObservationType());
  obs_type.public_info = false;
  obs_type.perfect_recall = false;
  obs_type.private_info = PrivateInfoType::kSinglePlayer;

  return std::make_shared<DurakObserver>(obs_type);
}

// -----------------------------------------------------------------------------
// DurakState implementation
// -----------------------------------------------------------------------------

DurakState::DurakState(std::shared_ptr<const Game> game, int rng_seed)
    : State(game) {
  // 1) Parse the deck from the game param "init_deck".
  const DurakGame* durak_game = down_cast<const DurakGame*>(game.get());
  std::string deck_str = durak_game->GetParameters().at("init_deck").string_value();

  deck_.reserve(kNumCards);
  {
    std::stringstream ss(deck_str);
    for (int i = 0; i < kNumCards; i++) {
      int c;
      char comma;
      ss >> c;
      deck_.push_back(c);
      if (i < kNumCards - 1) ss >> comma;  // consume the comma
    }
  }

  // 2) Initialize hands, table, discard, etc.
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
      // attacker wins
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
    // If deck is also empty => done
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

void DurakState::ApplyChanceAction(Action outcome) {
  // If we haven't dealt 6 cards each to both players, deal from top
  if (cards_dealt_ < kCardsPerPlayer * kNumPlayers) {
    // Deal this card to the next player
    int player_idx = cards_dealt_ % kNumPlayers;
    hands_[player_idx].push_back(outcome);
    ++deck_pos_;
    ++cards_dealt_;
  } else {
    // Reveal the last card as trump
    trump_card_ = deck_.back();
    trump_suit_ = SuitOf(deck_.back());
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
  if (game_over_) return;

  Player player = CurrentPlayer();
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

  auto &hand = hands_[player];
  auto it = std::find(hand.begin(), hand.end(), move);
  if (it == hand.end()) {
    return;  // invalid
  }

  if ((phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional) &&
      (player == attacker_)) {
    hand.erase(it);
    table_cards_.push_back(std::make_pair(move, -1));
    phase_ = RoundPhase::kAttack;
  } else if (phase_ == RoundPhase::kDefense && (player == defender_)) {
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

std::vector<Action> DurakState::LegalActions() const {
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

  if (phase_ == RoundPhase::kAttack || phase_ == RoundPhase::kAdditional) {
    if (player == attacker_) {
      if (table_cards_.empty()) {
        // can place any card
        for (int c : hand) {
          moves.push_back(c);
        }
      } else {
        // can only place ranks that appear on the table
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
      // can always FINISH_ATTACK if there's at least 1 card on the table
      if (!table_cards_.empty()) {
        moves.push_back(kExtraActionFinishAttack);
      }
    }
  } else if (phase_ == RoundPhase::kDefense) {
    // can TAKE_CARDS, or cover earliest uncovered, or FINISH_DEFENSE if none uncovered
    if (player == defender_) {
      bool any_uncovered = false;
      int earliest_uncovered = -1;
      for (int i = 0; i < static_cast<int>(table_cards_.size()); ++i) {
        if (table_cards_[i].second < 0) {
          any_uncovered = true;
          if (earliest_uncovered < 0) earliest_uncovered = i;
        }
      }
      if (!any_uncovered) {
        moves.push_back(kExtraActionFinishDefense);
      } else {
        moves.push_back(kExtraActionTakeCards);
      // try to cover earliest
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

  std::sort(moves.begin(), moves.end());
  return moves;
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
    return;
  }
  phase_ = RoundPhase::kDefense;
}

// The defender says "done" if all are covered; else effectively picks up
void DurakState::DefenderFinishesDefense() {
  // if uncovered => pick up, else discard
  bool uncovered = false;
  for (auto &pair : table_cards_) {
    if (pair.second < 0) {
      uncovered = true;
      break;
    }
  }
  if (uncovered) {
    // Takes cards
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

// Observations (unchanged, placeholders)
std::string DurakState::InformationStateString(Player player) const {
  // For single-player private info style, it's typically the same as
  // ObservationString if we are only exposing that player's hand.
  return ObservationString(player);
}

std::string DurakState::ObservationString(Player player) const {
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

void DurakState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ObservationTensor(player, values);
}

void DurakState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), 157);
  for (int i = 0; i < 157; i++) {
    values[i] = 0.f;
  }
}

// -----------------------------------------------------------------------------
// DurakObserver
// -----------------------------------------------------------------------------

DurakObserver::DurakObserver(IIGObservationType iig_obs_type)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      iig_obs_type_(iig_obs_type) {}

void DurakObserver::WriteTensor(const State& observed_state, int player,
                                Allocator* allocator) const {
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
  const DurakState& state = open_spiel::down_cast<const DurakState&>(observed_state);
  return state.ObservationString(player);
}

}  // namespace durak
}  // namespace open_spiel
