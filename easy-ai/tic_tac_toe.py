from easyAI import TwoPlayerGame, SSS, Negamax, DUAL, solve_with_depth_first_search, TranspositionTable
from easyAI.Player import Human_Player, AI_Player


class TicTacToe(TwoPlayerGame):
    """The board positions are numbered as follows:
    1 2 3
    4 5 6
    7 8 9
    """

    def __init__(self, players=None):
        self.players = players
        self.board = [0 for i in range(9)]
        self.current_player = 1  # player 1 starts.

    def possible_moves(self):
        return [i + 1 for i, e in enumerate(self.board) if e == 0]

    def make_move(self, move):
        self.board[int(move) - 1] = self.current_player

    def unmake_move(self, move):  # optional method (speeds up the AI)
        self.board[int(move) - 1] = 0

    def lose(self):
        """ Has the opponent "three in line ?" """
        return any(
            [
                all([(self.board[c - 1] == self.opponent_index) for c in line])
                for line in [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],  # horiz.
                    [1, 4, 7],
                    [2, 5, 8],
                    [3, 6, 9],  # vertical
                    [1, 5, 9],
                    [3, 5, 7],
                ]
            ]
        )  # diagonal

    def is_over(self):
        return (self.possible_moves() == []) or self.lose()

    def show(self):
        print(
            "\n"
            + "\n".join(
                [
                    " ".join([[".", "O", "X"][self.board[3 * j + i]]
                             for i in range(3)])
                    for j in range(3)
                ]
            )
        )

    def scoring(self):
        return -100 if self.lose() else 0

    def ttentry(self):
        return "".join([str(i) for i in self.board])


CONSOLE_WIDTH = 50


def get_padded_string(s):
    one_sided_padding = (CONSOLE_WIDTH - 2 - len(s))//2
    return '*' + ' ' * one_sided_padding + msg + ' ' * one_sided_padding + '*'


if __name__ == "__main__":

    from easyAI import AI_Player, Negamax

    ai_algo = Negamax(6)

    # game = TicTacToe([Human_Player(), AI_Player(ai_algo)])
    # scoring = 100
    # game.play()

    # 2. Dodatni scoring sprawia, że algorytm nie chce wygrać (wybiera ruchy jak najlepsze dla gracza a nie dla siebie)
    # Skala scoringu nie ma tutaj znaczenia.

    # 3.

    did_anybody_win = {
        3: [],
        6: [],
        9: [],
    }

    for ai_foresight in [3, 6, 9]:
        print('-'*CONSOLE_WIDTH)
        msg = f'For depth {ai_foresight}'

        print(get_padded_string(msg))

        print('-'*CONSOLE_WIDTH)
        for ai_approach in [Negamax(ai_foresight), SSS(ai_foresight), DUAL(ai_foresight)]:
            print('\n\n')
            print('-'*CONSOLE_WIDTH)
            print(f'For :{ai_approach.__class__.__name__} ai:')

            player = AI_Player(ai_approach)
            game = TicTacToe([player, player])
            history = game.play()

            print(len(history))
            did_anybody_win[ai_foresight].append(
                len(history) != 9 + 1)  # 9 moves of both players + ending state

    
    print(f"Did anybody win for each depth: {did_anybody_win}")
    # Argumenty do konstruktora obiektów reprezentujących AI wyglądają następująco
    # depth: ile ruchów na planszy sprawdzić do przodu? dla parametru 9 ai zna całe drzewo stanów dla kółka i krzyżyk
    # depth 3 jest za mały żeby ai grało dobrze, 6 i 9 jest ok
    # scoring: ile punktow sie zdobywa za przegrana/wygrana - "motywator" dla algorytmu, można nie ustawiać, nieustawiony scoring jest pobierany z samej gry
    # win_score: próg ponad którym ai uznaje grę za wygraną, detal implementacyjny, można nie ustawiać
    # tt: tablica transpozycji, jak poniżej 
    # przeszukanie wyczerpujące wszystkie stany
    perfect_game_state_indicator = solve_with_depth_first_search(
        game=TicTacToe(),
        win_score=100
    )

    # print(f'Can always win? {r}')
    # print(f'At what depth? {d}')
    # print(f'First move: {m}')
    print(perfect_game_state_indicator) # 0 oznacza że można wymusić remis, nie można wygrać jeżeli ruchy obu graczy są optymalne
    assert perfect_game_state_indicator == 0 # zawsze bedzie true

    # https://zulko.github.io/easyAI/ai_descriptions.html

    # 4. Tablica transpozycji
    # Cachuje wszystkie ruchy ktore zostaly juz raz rozstrzygniete przez AI, zmniejsza naklad obliczeniowy
    # (dla danego stanu liczymy nastepne ruchy tylko raz)
    ttable = TranspositionTable()
    # do tego trzeba zaimplementować dla gry metode .ttentry(), która zwraca klucz do hashowania
    # tak się tego używa
    ai = Negamax(9, tt=ttable)
    player = AI_Player(ai)
    game = TicTacToe(players=[player, player])
    game.play()

    ttable.to_json_file('transposition_table.json')
    # zapistanie do pliku (mozna pozniej zaimportować i wznowić grę)