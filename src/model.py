import numpy as np
import pandas as pd
from math import log
from typing import List, Dict

VERBOSE = False


def psi(e: float, b: int = 10, c: int = 3) -> float:
    return c * log(1 + e, b)


class Team:
    def __init__(self, ident: str):
        self.id: str = ident
        self.season: int = 0
        self.goal_home_abs: int = 0
        self.goal_away_abs: int = 0
        self.n_games: int = 1
        self.R_home: float = 0.
        self.R_away: float = 0.

    def get_goal_avg(self) -> (float, float):
        return self.goal_home_abs / self.n_games, self.goal_away_abs / self.n_games

    def expected_goal_diff(self, home: bool = True, b: int = 10, c: int = 3) -> float:
        rating = self.R_home if home else self.R_away
        norm = 1 if rating >= 0 else -1
        return norm * (b ** (abs(rating) / c) - 1)


class League:
    def __init__(self, ident: str):
        self.id: str = ident
        self.teams: Dict[str, Team] = {}
        self.season: int = 0

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
        for t in ['H', 'A']:
            team = self.teams[inc[t + 'ID']]
            team.n_games += 1
            if t == 'H':
                team.goal_home_abs += inc['HSC']
            else:
                team.goal_away_abs += inc['ASC']

    def get_goal_avg(self, hid, aid):
        h_avg = 0
        a_avg = 0
        if hid in self.teams:
            h_avg, _ = self.teams[hid].get_goal_avg()
        if aid in self.teams:
            _, a_avg = self.teams[aid].get_goal_avg()
        return h_avg, a_avg

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
            if today_games['LID'].iloc[i] not in self.leagues:
                continue
            league = self.leagues[today_games['LID'].iloc[i]]
            pred_outcome = league.predict_match_outcome(today_games['HID'].iloc[i], today_games['AID'].iloc[i])
            diff = pred_outcome
            if diff > 0.6:  # win home
                odds = 0
            elif diff < -0.8:  # win away
                odds = 2
            else:
                odds = 1

            bets[i, odds] = max(min_bet, self.bankroll / 50)
        if VERBOSE:
            print('Opportunities: {}'.format(N))
        return pd.DataFrame(data=bets, columns=['BetH', 'BetD', 'BetA'], index=today_games.index)

    def update_data(self, inc: pd.DataFrame):
        if inc.empty:
            return

        self.data = inc if self.data.empty else self.data.append(inc)

        for _, row in inc.iterrows():
            if row['Sea'] != 2005:
                continue
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


if __name__ == "__main__":
    lr = 0.1
    gamma = 0.3
    a_team = Team(1)
    b_team = Team(2)
    a_team.R_home = 1.6
    a_team.R_away = 0.4
    b_team.R_home = 0.3
    b_team.R_away = -1.2
    print(a_team.expected_goal_diff(home=True))
    print(b_team.expected_goal_diff(home=False))
    exp_dif = a_team.expected_goal_diff(home=True) - b_team.expected_goal_diff(home=False)
    real_dif = 4 - 1
    error = abs(real_dif - exp_dif)
    ps = psi(error)
    ps_h = ps if exp_dif < real_dif else -ps
    ps_a = ps if exp_dif > real_dif else -ps

    rah_hat = a_team.R_home + ps_h * lr
    a_team.R_away = a_team.R_away + (rah_hat - a_team.R_home) * gamma
    a_team.R_home = rah_hat

    rba_hat = b_team.R_away + ps_a * lr
    b_team.R_home = b_team.R_home + (rba_hat - b_team.R_away) * gamma
    b_team.R_away = rba_hat
    print()
