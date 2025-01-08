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

constexpr int kNumPlayers = 2;
constexpr int kNumCards = 36;          // 9 ranks (6..A) * 4 suits
constexpr int kCardsPerPlayer = 6;
constexpr int kExtraActionTakeCards      = kNumCards;     // 36
constexpr int kExtraActionFinishAttack   = kNumCards + 1; // 37
constexpr int kExtraActionFinishDefense  = kNumCards + 2; // 38

enum class RoundPhase {
  kChance = 0,
  kAttack = 1,
  kDefense = 2,
  kAdditional = 3
};

// Helper functions to interpret 0..35 as card indices.
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
  explicit DurakState(std::shared_ptr<const Game> game, int rng_seed);
  DurakState(const DurakState&) = default;
  DurakState& operator=(const DurakState&) = delete;

  // Core API
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

  // Game state
  std::vector<int> deck_;  
  std::array<std::vector<int>, kNumPlayers> hands_;
  std::vector<std::pair<int, int>> table_cards_;
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
// DurakGame
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

  // Implement deck shuffling.
  void ShuffleDeck(std::mt19937* rng, std::vector<int> deck, int begin, int end);

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

 private:
  mutable int rng_seed_ = 0;
};

// -----------------------------------------------------------------------------
// An Observer
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
