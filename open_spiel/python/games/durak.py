# Copyright 2025
# Based on the MIT-licensed Durak implementation by Manuel Boesl and Borislav Radev
# https://github.com/ManuelBoesl/durak_card_game

import enum
import numpy as np
import pyspiel
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------------------
# Global definitions and constants
# ----------------------------------------------------------------------------------

_NUM_PLAYERS = 2
_NUM_CARDS = 36  # 6..10, J, Q, K, A in four suits => 9 ranks * 4 suits
_CARDS_PER_PLAYER = 6  # Each player is dealt (up to) 6 cards
# We'll define a 36-card deck enumerated as 0..35.
# suit = card // 9, rank = card % 9
# Suits (0..3): 0=♠, 1=♣, 2=♦, 3=♥  (just for printing)
# Ranks (0..8): [6, 7, 8, 9, 10, J, Q, K, A]

_DECK = list(range(_NUM_CARDS))


def suit_of(card: int) -> int:
    return card // 9

def rank_of(card: int) -> int:
    return card % 9

def card_to_string(card: int) -> str:
    """Convert card index (0..35) to human-readable form, e.g. '10♥'."""
    if card < 0 or card >= _NUM_CARDS:
        return "None"
    # Suit order: 0=♠, 1=♣, 2=♦, 3=♥
    suit_symbols = ["♠", "♣", "♦", "♥"]
    rank_symbols = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    s = suit_of(card)
    r = rank_of(card)
    return f"{rank_symbols[r]}{suit_symbols[s]}"

def card_to_action(card: int) -> int:
    """Actions [0..35] correspond to playing 'card' from hand."""
    return card

def action_to_card(action: int) -> Optional[int]:
    """If the action is in [0..35], interpret as that card index; else None."""
    if 0 <= action < _NUM_CARDS:
        return action
    return None


# Additional actions beyond card indices:
#   TAKE_CARDS: defender picks up all the cards on the table
#   FINISH_ATTACK: attacker stops adding any more attacking cards
#   FINISH_DEFENSE: defender indicates they are done defending if all covered
class ExtraAction(enum.IntEnum):
    TAKE_CARDS = _NUM_CARDS
    FINISH_ATTACK = _NUM_CARDS + 1
    FINISH_DEFENSE = _NUM_CARDS + 2


class RoundPhase(enum.IntEnum):
    """Phases of a Durak round."""
    CHANCE = 0       # Dealing initial cards (and revealing trump)
    ATTACK = 1       # Attacker can place multiple cards before finishing
    DEFENSE = 2      # Defender tries to cover them
    ADDITIONAL = 3   # Attacker can add more cards if all are covered so far


_GAME_TYPE = pyspiel.GameType(
    short_name="python_durak",
    long_name="Python Durak",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CARDS + 3,  # 36 card-plays + TAKE_CARDS + 2 finishes
    max_chance_outcomes=_NUM_CARDS,       # any card in the deck as a chance outcome
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=300  # a safe upper bound
)


class DurakGame(pyspiel.Game):
    """The Durak game definition for OpenSpiel."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        return DurakState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return DurakObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params
        )


class DurakState(pyspiel.State):
    """OpenSpiel state for Durak with multi-card attacking."""

    def __init__(self, game: DurakGame):
        super().__init__(game)
        self._deck = _DECK.copy()
        np.random.shuffle(self._deck)

        # Each player's hand
        self._hands: List[List[int]] = [[] for _ in range(_NUM_PLAYERS)]

        # The table: list of (attacking_card, defending_card_or_None)
        self._table_cards: List[Tuple[int, Optional[int]]] = []

        # Discard pile for covered cards
        self._discard: List[int] = []

        # Trump suit and trump card
        self._trump_suit: Optional[int] = None
        self._trump_card: Optional[int] = None

        # For dealing the initial 6 cards each + revealing trump
        self._cards_dealt = 0
        self._deck_pos = 0

        # Roles
        self._attacker = 0
        self._defender = 1
        self._phase = RoundPhase.CHANCE
        self._round_starter = 0  # Who began the current round as attacker?

        self._game_over = False

    # --------------------------------------------------------------------------
    # OpenSpiel API
    # --------------------------------------------------------------------------
    def current_player(self) -> int:
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase == RoundPhase.CHANCE:
            return pyspiel.PlayerId.CHANCE
        if self._phase in [RoundPhase.ATTACK, RoundPhase.ADDITIONAL]:
            return self._attacker
        return self._defender  # RoundPhase.DEFENSE => defender

    def is_terminal(self) -> bool:
        return self._game_over

    def returns(self) -> List[float]:
        if not self._game_over:
            return [0.0, 0.0]
        # If exactly one player has cards, that player is the loser => -1,
        # the other is the winner => +1. 
        # If both have cards or none does, see special rule below.

        players_with_cards = [p for p in range(_NUM_PLAYERS) if len(self._hands[p]) > 0]
        if len(players_with_cards) == 1:
            # Exactly one has cards => that one is the loser
            loser = players_with_cards[0]
            result = [0.0, 0.0]
            result[loser] = -1.0
            result[1 - loser] = 1.0
            return result

        if len(players_with_cards) == 2:
            # If both have cards, it's a corner-case scenario. We could treat it as a draw or keep going.
            # Typically should not happen with correct logic, but just in case, we do [0,0].
            return [0.0, 0.0]

        # Now, if len(players_with_cards) == 0 => neither has cards.
        # We check if the deck is also empty. 
        # If the deck is empty, the attacker of that last round is the winner by the user's request.
        # If the deck is not empty, that means we ended a round successfully (defender survived), and
        # we should typically continue playing. But if it's truly terminal, let's follow the user's
        # instruction: "both are empty but deck isn't => next round." For safety, if it's forced terminal,
        # just treat it as a "defender's success." We'll do something consistent:
        if self._deck_pos >= len(self._deck):
            # All deck is used. So attacker is declared winner.
            result = [0.0, 0.0]
            atk = self._attacker  # or we could store the round starter
            # The user specifically wants the *attacker of the last round* to be the winner
            result[atk] = 1.0
            result[1 - atk] = -1.0
            return result
        else:
            # Deck not empty => if this is terminal, we'll consider that the defender survived => defender = winner
            # (But in real gameplay we'd refill. The user wants to keep going. 
            # We'll do a minor fallback: treat it as 0,0. 
            # Uncomment below if you want to treat the attacker or defender as winner in this edge scenario.
            """
            # Option A: declare the defender is winner
            defd = self._defender
            result = [0.0, 0.0]
            result[defd] = 1.0
            result[1 - defd] = -1.0
            return result
            """
            return [0.0, 0.0]

    def __str__(self) -> str:
        if self._phase == RoundPhase.CHANCE:
            return (f"Chance node: dealing... cards_dealt={self._cards_dealt}, "
                    f"trump_suit={self._trump_suit if self._trump_suit is not None else '??'}")

        lines = []
        lines.append(f"Attacker={self._attacker}, Defender={self._defender}")
        lines.append(f"Phase={RoundPhase(self._phase).name}, Discarded={len(self._discard)}, DeckRemaining={len(self._deck)-self._deck_pos}")
        if self._trump_card is not None:
            lines.append(f"Trump={card_to_string(self._trump_card)} (suit={self._trump_suit})")

        for p in range(_NUM_PLAYERS):
            hand_str = [card_to_string(c) for c in self._hands[p]]
            lines.append(f"Player {p} hand: {hand_str}")

        table_str = []
        for (ac, dc) in self._table_cards:
            if dc is None:
                table_str.append(f"{card_to_string(ac)}->?")
            else:
                table_str.append(f"{card_to_string(ac)}->{card_to_string(dc)}")
        lines.append("Table: " + ", ".join(table_str))
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Chance Node Logic (dealing & revealing trump)
    # --------------------------------------------------------------------------
    def is_chance_node(self) -> bool:
        return (self._phase == RoundPhase.CHANCE)

    def chance_outcomes(self):
        # The first 12 outcomes deal 6 cards each to attacker/defender.
        # Then 1 outcome reveals the bottom card as trump.
        # Probability for each is 1.0 in a "deterministic" sense once we fix the deck order.

        if self._cards_dealt < _CARDS_PER_PLAYER * _NUM_PLAYERS:
            # We are dealing from top
            next_card = self._deck[self._deck_pos]
            return [(next_card, 1.0)]
        else:
            # Next step is to reveal the bottom card as trump
            if self._trump_card is None:
                bottom_card = self._deck[-1]
                return [(bottom_card, 1.0)]
            return []

    def _apply_chance_action(self, outcome: int):
        if self._cards_dealt < _CARDS_PER_PLAYER * _NUM_PLAYERS:
            # Dealing a top card to the next player
            player_idx = self._cards_dealt % _NUM_PLAYERS
            self._hands[player_idx].append(outcome)
            self._deck_pos += 1
            self._cards_dealt += 1
        else:
            # Reveal the bottom card as trump
            self._trump_card = outcome
            self._trump_suit = suit_of(outcome)
            self._decide_first_attacker()
            self._phase = RoundPhase.ATTACK
            self._round_starter = self._attacker

    # --------------------------------------------------------------------------
    # Move logic
    # --------------------------------------------------------------------------
    def _legal_actions(self, player: int):
        if self._game_over or self.is_chance_node():
            return []

        # If the phase is ATTACK or ADDITIONAL, the attacker can keep playing new attack cards
        # or can do FINISH_ATTACK.
        # If the phase is DEFENSE, the defender can choose to cover an uncovered card (play a card)
        # or TAKE_CARDS or FINISH_DEFENSE if everything is covered.

        actions = []
        hand = self._hands[player]

        if self._phase in [RoundPhase.ATTACK, RoundPhase.ADDITIONAL] and (player == self._attacker):
            # 1) The attacker can place any card that is "legal" for multi-attack:
            #    no rank restriction for the very first card(s)? In some Durak variants,
            #    the attacker can only place ranks that already appear on the table, but
            #    we want them to be free to place any card as we do multi-attack. We'll
            #    match the original user note: "As long as the rank is among the ranks
            #    on the table, or if no table cards yet, you can put any card."
            if len(self._table_cards) == 0:
                # No cards on the table yet, so only card plays are allowed. 
                for c in hand:
                    actions.append(card_to_action(c))
            elif len(self._table_cards) < _CARDS_PER_PLAYER and len(self._hands[self._defender]):
                # The attacker can toss in cards whose rank matches any rank on the table
                # Can't attack if there are too many cards on the table:
                #   - either if there are already as many cards as the defender has
                #   - or if 6 cards have been already played
                ranks_on_table = set(rank_of(ac) for (ac, dc) in self._table_cards)
                ranks_on_table.update(rank_of(dc) for (ac, dc) in self._table_cards if dc is not None)
                for c in hand:
                    if rank_of(c) in ranks_on_table:
                        actions.append(card_to_action(c))

            # 2) Allow FINISH_ATTACK only if there's at least one card on the table:
            if len(self._table_cards) > 0:
                actions.append(ExtraAction.FINISH_ATTACK)

        elif self._phase == RoundPhase.DEFENSE and (player == self._defender):
            # The defender can do:
            #  - TAKE_CARDS at any time
            #  - If there's at least one uncovered card, attempt to cover with a valid card
            #  - If all are covered, FINISH_DEFENSE
            uncovered = [(i, ac) for i, (ac, dc) in enumerate(self._table_cards) if dc is None]
            if len(uncovered) == 0:
                # everything is covered => can FINISH_DEFENSE
                actions.append(ExtraAction.FINISH_DEFENSE)
            else:
                # can TAKE_CARDS
                actions.append(ExtraAction.TAKE_CARDS)
                # or try to defend each uncovered card
                for i, att_card in uncovered:
                    # We'll do a "one uncovered card at a time" approach:
                    # The next user play of c means "defend the earliest uncovered".
                    # So effectively we only allow covering the first uncovered card in the list.
                    # If you want to allow the defender to pick *which* uncovered to defend,
                    # you'd have to define an action indexing i. Here we do a simpler approach:
                    # we only let them cover the earliest uncovered. That was also how the
                    # original single-step logic worked. 
                    # So let's do that:
                    # earliest index
                    if i == uncovered[0][0]:
                        for c in hand:
                            if self._can_defend_card(c, att_card):
                                actions.append(card_to_action(c))
                # If you want the defender to be able to choose any uncovered card to defend first,
                # you'd need a more complex action representation. For now, we keep it simple.

        return sorted(actions)

    def _apply_action(self, action: int):
        if self.is_chance_node():
            self._apply_chance_action(action)
            return

        player = self.current_player()

        if action >= _NUM_CARDS:
            # Extra action
            if action == ExtraAction.TAKE_CARDS:
                self._defender_takes_cards()
            elif action == ExtraAction.FINISH_ATTACK:
                self._attacker_finishes_attack()
            elif action == ExtraAction.FINISH_DEFENSE:
                self._defender_finishes_defense()
            self._check_game_over()
            return

        # Otherwise, it's a card index
        card = action_to_card(action)
        if card is not None and card in self._hands[player]:
            if self._phase in [RoundPhase.ATTACK, RoundPhase.ADDITIONAL] and player == self._attacker:
                # Attacker places this card face up on the table (uncovered)
                self._hands[player].remove(card)
                self._table_cards.append((card, None))
                # We remain in ATTACK if we want to keep multi-attacking
                # The user can eventually do FINISH_ATTACK
                # The only difference is that if there's at least one uncovered card, the defender
                # might start thinking. But to keep the single-step approach, we don't transition yet.
                self._phase = RoundPhase.ATTACK

            elif self._phase == RoundPhase.DEFENSE and player == self._defender:
                # Defender tries to cover the earliest uncovered card
                uncovered = [(i, ac) for i, (ac, dc) in enumerate(self._table_cards) if dc is None]
                if uncovered:
                    earliest_idx, att_card = uncovered[0]
                    if self._can_defend_card(card, att_card):
                        self._hands[player].remove(card)
                        self._table_cards[earliest_idx] = (att_card, card)
                        # If we still have more uncovered to do, stay in DEFENSE
                        # If all are covered, we shift to ADDITIONAL
                        if all(dc is not None for (ac, dc) in self._table_cards):
                            self._phase = RoundPhase.ADDITIONAL

            # else: invalid usage => do nothing (shouldn't happen if we filter in legal_actions)

        self._check_game_over()

    # --------------------------------------------------------------------------
    # Utility methods
    # --------------------------------------------------------------------------
    def _decide_first_attacker(self):
        """Decide the first attacker based on the lowest trump card among each player's 6 cards."""
        lowest_trump = None
        who = 0
        for p in range(_NUM_PLAYERS):
            for c in self._hands[p]:
                if suit_of(c) == self._trump_suit:
                    if (lowest_trump is None) or (rank_of(c) < rank_of(lowest_trump)):
                        lowest_trump = c
                        who = p
        self._attacker = who
        self._defender = 1 - who

    def _can_defend_card(self, defense_card: int, attack_card: int) -> bool:
        att_s, att_r = suit_of(attack_card), rank_of(attack_card)
        def_s, def_r = suit_of(defense_card), rank_of(defense_card)

        # same suit, higher rank
        if att_s == def_s and def_r > att_r:
            return True
        # defend with trump if attack is not trump
        if def_s == self._trump_suit and att_s != self._trump_suit:
            return True
        # if both trump, rank must be higher
        if att_s == self._trump_suit and def_s == self._trump_suit and def_r > att_r:
            return True
        return False

    def _defender_takes_cards(self):
        """Defender picks up all table cards, round ends, same attacker remains."""
        for (ac, dc) in self._table_cards:
            self._hands[self._defender].append(ac)
            if dc is not None:
                self._hands[self._defender].append(dc)
        self._table_cards.clear()
        self._phase = RoundPhase.ATTACK
        self._refill_hands()

    def _attacker_finishes_attack(self):
        """Attacker decides not to lay more cards, so we transition to DEFENSE or check if all are covered."""
        if len(self._table_cards) == 0:
            # Strange corner case: no attacks at all. We stay attacker? 
            return
        self._phase = RoundPhase.DEFENSE

    def _defender_finishes_defense(self):
        """Defender indicates they are done if all are covered."""
        # If there's an uncovered card, that means the defender is effectively giving up => TAKE_CARDS logic.
        uncovered = any(dc is None for (ac, dc) in self._table_cards)
        if uncovered:
            self._defender_takes_cards()
        else:
            # Everything covered => move them to discard, roles swap, refill
            for (ac, dc) in self._table_cards:
                self._discard.append(ac)
                if dc is not None:
                    self._discard.append(dc)
            self._table_cards.clear()
            # swap roles
            old_attacker = self._attacker
            self._attacker = self._defender
            self._defender = old_attacker
            self._refill_hands()
            self._phase = RoundPhase.ATTACK

    def _refill_hands(self):
        """Refill to 6 cards: attacker first, then defender, from top of deck (excluding trump if at bottom)."""
        order = [self._attacker, self._defender]
        while self._deck_pos < len(self._deck):  # the last card is the trump (face up)
            for p in order:
                if len(self._hands[p]) < _CARDS_PER_PLAYER and self._deck_pos < len(self._deck):
                    c = self._deck[self._deck_pos]
                    self._deck_pos += 1
                    self._hands[p].append(c)
                # if deck is exhausted, break
            if all(len(self._hands[p]) >= _CARDS_PER_PLAYER for p in order):
                break

    def _check_game_over(self):
        """Check if the game ends now."""
        # If either player is out of cards => game over
        # If both are out and deck is empty => attacker wins per user request
        # If both are out but deck not empty => normal round end -> refill (should keep playing).
        # But if there's no next step for some reason, the game might end. We'll interpret gracefully.

        # 1) if a player has no cards:
        if (len(self._hands[0]) == 0 or len(self._hands[1]) == 0) and self._deck_pos >= len(self._deck):
            # Might be both or just one. We'll check the terminal returns logic in returns().
            self._game_over = True
            return

        # 2) if both players have no cards:
        if (len(self._hands[0]) == 0 and len(self._hands[1]) == 0):
            # If deck is also empty => attacker is winner => terminal
            if self._deck_pos >= len(self._deck):
                self._game_over = True
                return
            else:
                # Deck not empty => that means we ended a round, so normally we'd refill and continue
                # We'll do that:
                self._refill_hands()
                # If for some reason it doesn't produce new cards for either, the game might stall,
                # but let's see. We won't forcibly end it. 
                # We'll keep going. If it hits a stall, the returns logic might see both with cards eventually or not.
                pass


# ----------------------------------------------------------------------------------
# Observer
# ----------------------------------------------------------------------------------

class DurakObserver:
    """Observer for Durak, following the PyObserver interface."""

    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        self._iig_obs_type = iig_obs_type

        pieces = [
            ("player", _NUM_PLAYERS, (_NUM_PLAYERS,)),
            ("trump_suit", 4, (4,)),   # one-hot
            ("phase", 4, (4,)),        # one-hot for RoundPhase
            ("deck_size", 1, (1,)),
            ("attacker_ind", 1, (1,)),
            ("defender_ind", 1, (1,)),
            ("trump_card", _NUM_CARDS, (_NUM_CARDS,)),
        ]

        # If single-player private info, we encode player's hand
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("my_cards", _NUM_CARDS, (_NUM_CARDS,)))

        # If public info is True, we reveal the table layout
        if iig_obs_type.public_info:
            pieces.append(("table_attack", _NUM_CARDS, (_NUM_CARDS,)))
            pieces.append(("table_defense", _NUM_CARDS, (_NUM_CARDS,)))

        total_size = sum(sz for _, sz, _ in pieces)
        self.tensor = np.zeros(total_size, np.float32)
        self.dict = {}
        idx = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[idx : idx + size].reshape(shape)
            idx += size

    def set_from(self, state: DurakState, player: int):
        self.tensor.fill(0.0)

        # 1) player indicator
        if "player" in self.dict:
            self.dict["player"][player] = 1

        # 2) trump_suit
        if state._trump_suit is not None and "trump_suit" in self.dict:
            self.dict["trump_suit"][state._trump_suit] = 1

        # 3) trump_card
        if "trump_card" in self.dict and state._trump_card is not None:
            self.dict["trump_card"][state._trump_card] = 1

        # 4) phase
        if "phase" in self.dict:
            self.dict["phase"][state._phase] = 1

        # 5) deck_size
        if "deck_size" in self.dict:
            ds = (len(state._deck) - state._deck_pos)
            self.dict["deck_size"][0] = ds / float(_NUM_CARDS)

        # 6) attacker_ind, defender_ind
        if "attacker_ind" in self.dict:
            self.dict["attacker_ind"][0] = float(player == state._attacker)
        if "defender_ind" in self.dict:
            self.dict["defender_ind"][0] = float(player == state._defender)

        # 7) my_cards
        if "my_cards" in self.dict:
            for c in state._hands[player]:
                self.dict["my_cards"][c] = 1

        # 8) table_attack, table_defense
        if "table_attack" in self.dict and "table_defense" in self.dict:
            for (ac, dc) in state._table_cards:
                self.dict["table_attack"][ac] = 1
                if dc is not None:
                    self.dict["table_defense"][dc] = 1

    def string_from(self, state: DurakState, player: int) -> str:
        lines = []
        lines.append(f"Player {player} viewpoint")
        lines.append(f"Phase: {RoundPhase(state._phase).name}")
        if state._trump_card is not None:
            lines.append(f"Trump card: {card_to_string(state._trump_card)}") 
        lines.append(f"Attacker={state._attacker}, Defender={state._defender}")
        # My hand
        lines.append(f"My hand: {[card_to_string(c) for c in sorted(state._hands[player])]}")

        # Table
        table_str = []
        for (ac, dc) in state._table_cards:
            if dc is None:
                table_str.append(f"{card_to_string(ac)}->?")
            else:
                table_str.append(f"{card_to_string(ac)}->{card_to_string(dc)}")
        lines.append(f"Table: {table_str}")
        lines.append(f"DeckRemaining: {len(state._deck)-state._deck_pos}")
        return " | ".join(lines)


pyspiel.register_game(_GAME_TYPE, DurakGame)
