// Copyright 2025
// Based on the MIT-licensed Durak implementation by Manuel Boesl and Borislav Radev
// and your Python translation in OpenSpiel.
//
// Durak is a turn-based card game for 2 players with imperfect information.
//
// This is the header file (durak.h) for the C++ implementation in OpenSpiel.

#ifndef OPEN_SPIEL_GAMES_DURAK_H_
#define OPEN_SPIEL_GAMES_DURAK_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace durak {

// -----------------------------------------------------------------------------
// Global definitions and constants
// -----------------------------------------------------------------------------

// Basic game settings for 2-player Durak.
constexpr int kNumPlayers = 2;
constexpr int kNumCards = 36;          // 9 ranks (6..A) * 4 suits
constexpr int kCardsPerPlayer = 6;     // Each player is refilled up to 6 cards
constexpr int kExtraActionTakeCards      = kNumCards;     // 36
constexpr int kExtraActionFinishAttack   = kNumCards + 1; // 37
constexpr int kExtraActionFinishDefense  = kNumCards + 2; // 38

// RoundPhase enumerates the high-level flow of the game.
enum class RoundPhase {
  kChance = 0,      // Dealing initial cards (and revealing trump)
  kAttack = 1,      // Attacker(s) placing cards
  kDefense = 2,     // Defender trying to cover
  kAdditional = 3   // Attacker can add more cards after all current ones covered
};

// Helper inline functions to get suit/rank from a 0..35 card index.
inline int SuitOf(int card) { return card / 9; }
inline int RankOf(int card) { return card % 9; }

// A small helper for debugging/logging if needed.
inline std::string CardToString(int card) {
  if (card < 0 || card >= kNumCards) return "None";
  static const std::array<const char*, 4> suit_symbols = {"♠", "♣", "♦", "♥"};
  static const std::array<const char*, 9> rank_symbols =
      {"6", "7", "8", "9", "10", "J", "Q", "K", "A"};
  int s = SuitOf(card);
  int r = RankOf(card);
  return std::string(rank_symbols[r]) + suit_symbols[s];
}

// Forward declarations
class DurakGame;
class DurakObserver;

// -----------------------------------------------------------------------------
// DurakState: the game state container & logic
// -----------------------------------------------------------------------------

class DurakState : public State {
 public:
  explicit DurakState(std::shared_ptr<const Game> game);
  DurakState(const DurakState&) = default;
  DurakState& operator=(const DurakState&) = delete;

  // Core API from open_spiel::State
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ToString() const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;

  // Chance handling
  bool IsChanceNode() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Observations
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

 protected:
  // ApplyAction is the main method that modifies the state in response to moves.
  void DoApplyAction(Action move) override;

 private:
  // ---------- Helper/Private Methods ----------

  // For dealing chance cards and revealing the trump.
  void ApplyChanceAction(Action outcome);

  // Determines the first attacker based on the lowest trump in any player's hand.
  void DecideFirstAttacker();

  // Helpers for valid defense coverage, picking up cards, finishing phases, etc.
  bool CanDefendCard(int defense_card, int attack_card) const;
  void DefenderTakesCards();
  void AttackerFinishesAttack();
  void DefenderFinishesDefense();
  void RefillHands();
  void CheckGameOver();

  // ---------- Game State Members ----------

  // The deck of 36 cards. We store the order (top to bottom). The bottom card is
  // the trump reveal in classical Durak, but we keep it in the deck until reveal.
  std::vector<int> deck_;

  // Each player's hand of card indices.
  std::array<std::vector<int>, kNumPlayers> hands_;

  // The table: each element is (attacking_card, defending_card_or_null).
  std::vector<std::pair<int, int>> table_cards_;  // If defending_card is -1 => None

  // Discard pile for fully-covered cards.
  std::vector<int> discard_;

  // Which suit is trump? 0..3, or -1 if unknown.
  int trump_suit_ = -1;
  // The actual trump card index, or -1 if not revealed yet.
  int trump_card_ = -1;

  // Dealing progress: how many total cards have been dealt so far?
  int cards_dealt_ = 0;
  // Deck position for the next card to be dealt from the top. 
  int deck_pos_ = 0;

  // Roles
  Player attacker_ = 0;
  Player defender_ = 1;
  // Which phase are we in?
  RoundPhase phase_ = RoundPhase::kChance;

  // For reference or special rules: who started this round as attacker?
  Player round_starter_ = 0;

  // Game over flag
  bool game_over_ = false;
};

// -----------------------------------------------------------------------------
// DurakGame: the factory and top-level game object
// -----------------------------------------------------------------------------

class DurakGame : public Game {
 public:
  explicit DurakGame(const GameParameters& params);

  // Implement the base interface:
  int NumDistinctActions() const override { return kNumCards + 3; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return kNumCards; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1.0; }
  double MaxUtility() const override { return 1.0; }
  absl::optional<double> UtilitySum() const override { return 0.0; }

  // For Durak, a safe upper bound on game length could be fairly high.
  int MaxGameLength() const override { return 300; }

  // We do have up to 12 dealing moves for the initial cards, plus 1 for trump reveal,
  // so max chance nodes might be 13 for the initial plus a few refills. But 36 is also safe.
  int MaxChanceNodesInHistory() const override { return 36; }

  // Observations
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;

  // Construct an observer that knows how to interpret states (see durak.cc).
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
};

// -----------------------------------------------------------------------------
// An Observer that can produce both a string and a tensor, using iig_obs_type
// to decide which pieces of private/public information to expose.
// -----------------------------------------------------------------------------

class DurakObserver : public Observer {
 public:
  explicit DurakObserver(IIGObservationType iig_obs_type);

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override;

  std::string StringFrom(const State& observed_state,
                         int player) const override;

 private:
  IIGObservationType iig_obs_type_;
};

}  // namespace durak
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DURAK_H_
