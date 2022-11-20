from easyAI import TwoPlayerGame,solve_with_iterative_deepening, Human_Player, AI_Player, Negamax


from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class LastCoin(TwoPlayerGame):
    """ In turn, the players remove one, two or three bones from a
    pile of bones. The player who removes the last bone loses. """

    def __init__(self, players=None):
        self.players = players
        self.num_coins = 20
        self.current_player = 1

    def possible_moves(self):
        return ['1', '2', '3', '4']

    def make_move(self, move):
        self.num_coins -= int(move)

    def is_over(self):
        return self.num_coins <= 0

    def show(self):
        print(self.num_coins, 'monet pozostalo na stosie')

    def scoring(self):
        return 100 if self.is_over() else 0


if __name__ == '__main__':
    ai = Negamax(5)
    game = LastCoin([Human_Player(), AI_Player(ai)])
    # history = game.play()
    
    # print(history)
    
    # for i, j in history:
    #     print(f'Pobrano ze stosu {j} monet')
    r,d,m = solve_with_iterative_deepening(
    game=LastCoin(),
    ai_depths=range(2,10),
    win_score=100
    )

    print(f'Can always win? {r}')
    print(f'At what depth? {d}')
    print(f'First move: {m}')


    # 1.
    print(f'Check if that is correct...')
    ai = Negamax(int(m))
    player = AI_Player(ai)

    for _ in range(3):
        game = LastCoin([player, player])
        history = game.play()

        # Assert that first player won the game:
        assert history[-1].current_player == 1

        # Oba ai uzywaja ruchu 1 az do momentu, gdy gracz nr 1 moze doprowadzic do sytuacji ze wystepuje 6 monet
        # wtedy gracz nr 2 nie ma dobregu ruchu (może wywołać sytuacje gdzie na stole jest od  5 do 2 monet).
        # Taka pozycja kończy się zwycięstwem dla gracza nr 1.
    
    