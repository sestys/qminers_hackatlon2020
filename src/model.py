import numpy as np
import pandas as pd
from math import log
from typing import List, Dict


def psi(e: float, b: int = 10, c: int = 3) -> float:
    return c * log(1 + e, b)


class Team:
    def __init__(self, ident: str):
        self.id: str = ident
        self.season: int = 0
        self.n_games: int = 1
        self.R_home: float = 0.
        self.R_away: float = 0.

    def expected_goal_diff(self, home: bool = True, b: int = 10, c: int = 3) -> float:
        rating = self.R_home if home else self.R_away
        norm = 1 if rating >= 0 else -1
        return norm * (b ** (abs(rating) / c) - 1)


class League:
    def __init__(self, ident: str):
        self.id: str = ident
        self.teams: Dict[str, Team] = {}
        self.season: int = 0
        self.n_games: int = 0

    def add_team_if_missing(self, ident: str, team: Team = None):
        if ident in self.teams:
            return
        if team is None:
            team = Team(ident)

        self.teams[ident] = team

    def revise_pi_rating(self, hid, aid, hsc, asc, lr=0.1, gamma=0.3):
        exp_dif = self.predict_match_outcome(hid, aid)
        real_dif = hsc - asc
        error = abs(real_dif - exp_dif)
        ps = psi(error)
        ps_h = ps if exp_dif < real_dif else -ps
        ps_a = ps if exp_dif > real_dif else -ps
        a_team = self.teams[hid]
        b_team = self.teams[aid]
        rah_hat = a_team.R_home + ps_h * lr
        a_team.R_away = a_team.R_away + (rah_hat - a_team.R_home) * gamma
        a_team.R_home = rah_hat

        rba_hat = b_team.R_away + ps_a * lr
        b_team.R_home = b_team.R_home + (rba_hat - b_team.R_away) * gamma
        b_team.R_away = rba_hat

    def update_data(self, inc):
        hid = inc['HID']
        aid = inc['AID']
        self.revise_pi_rating(hid, aid, inc['HSC'], inc['ASC'])
        self.teams[hid].n_games += 1
        self.teams[aid].n_games += 1
        self.n_games += 1

    def predict_match_outcome(self, hid, aid):
        self.add_team_if_missing(hid)
        self.add_team_if_missing(aid)
        h_team = self.teams[hid]
        a_team = self.teams[aid]
        return h_team.expected_goal_diff(home=True) - a_team.expected_goal_diff(home=False)

    def __str__(self):
        teams = list(self.teams.keys())
        teams_str = ', '.join(str(t) for t in teams)
        return 'League: {}, season: {}, teams {}: {}'.format(self.id, self.season, len(teams), teams_str)


class Model:
    def __init__(self):
        self.bankroll: float = 0.0
        self.bankroll_variance = [float('inf'), float('-inf')]
        self.leagues: Dict[str, League] = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.odds_cols = ['OddsH', 'OddsD', 'OddsA']
        self.score_cols = ['HSC', 'ASC']
        self.team_cols = ['HID', 'AID']
        self.res_cols = ['H', 'D', 'A']

    def update_bankroll(self, curr):
        self.bankroll = curr
        self.bankroll_variance[0] = min(self.bankroll_variance[0], self.bankroll)
        self.bankroll_variance[1] = max(self.bankroll_variance[1], self.bankroll)

    def place_bets(self, opps: pd.DataFrame, summary: pd.DataFrame, inc: pd.DataFrame):
        self.update_data(inc)
        _summary = summary.iloc[0].to_dict()
        self.update_bankroll(_summary['Bankroll'])
        min_bet = _summary['Min_bet']
        max_bet = _summary['Max_bet']
        date = _summary['Date']
        today_games = opps[opps['Date'] == date]
        N = len(today_games)
        bets = np.zeros((N, 3))
        for i in range(N):
            game = today_games.iloc[i]
            if game['LID'] not in self.leagues or self.leagues[game['LID']].n_games < 1000:
                continue
            league = self.leagues[game['LID']]
            pred_outcome = league.predict_match_outcome(game['HID'], game['AID'])
            diff = pred_outcome
            bet = max(min_bet, min(max_bet, self.bankroll / 100))
            if diff > 1.:  # win home
                odds = 0
            elif diff < -1.3:  # win away
                odds = 2
            elif -0.4 < diff < 0.5:
                odds = 1
            else:
                odds = 0
                bet = 0

            bets[i, odds] = bet
        if VERBOSE:
            print('Opportunities: {}'.format(N))
        return pd.DataFrame(data=bets, columns=['BetH', 'BetD', 'BetA'], index=today_games.index)

    def update_data(self, inc: pd.DataFrame):
        if inc.empty:
            return
        self.data = inc if self.data.empty else self.data.append(inc)
        for _, row in inc.iterrows():
            if row['LID'] in self.leagues:
                league = self.leagues[row['LID']]
            else:
                league = League(row['LID'])
                self.leagues[row['LID']] = league
            league.update_data(row)

    def print_leagues(self):
        print(self.bankroll_variance)
        for _, league in self.leagues.items():
            print(league)
