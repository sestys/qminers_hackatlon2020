import numpy as np
import pandas as pd
from typing import List, Dict


class Model:
    def __init__(self):
        self.bankroll: float = 0.0
        self.odds_cols = ['OddsH', 'OddsD', 'OddsA']
        self.score_cols = ['HSC', 'ASC']
        self.res_cols = ['H', 'D', 'A']

    def place_bets(self, opps: pd.DataFrame, summary: pd.DataFrame, inc: pd.DataFrame):
        _summary = summary.iloc[0].to_dict()
        self.bankroll = _summary['Bankroll']
        min_bet = _summary['Min_bet']
        max_bet = _summary['Max_bet']
        date = _summary['Date']
        today_games = opps[opps['Date'] == date]
        N = len(today_games)
        odds = np.argmax(today_games[self.odds_cols].to_numpy(), axis=1)
        bets = np.zeros((N, 3))
        bets[np.arange(N), odds] = min_bet
        print('Opportunities: {}'.format(N))
        return pd.DataFrame(data=bets, columns=['BetH', 'BetD', 'BetA'], index=today_games.index)

