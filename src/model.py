import numpy as np
import pandas as pd
from typing import List, Dict

VERBOSE = False


class Team:
    def __init__(self, ident: str):
        self.id: str = ident
        self.season: int = 0
        self.goal_home_abs: int = 0
        self.goal_away_abs: int = 0.
        self.n_games: int = 1

    def get_goal_avg(self) -> (float, float):
        return self.goal_home_abs / self.n_games, self.goal_away_abs / self.n_games


class League:
    def __init__(self, ident: str):
        self.id: str = ident
        self.teams: Dict[str, Team] = {}
        self.season: int = 0

    def add_team(self, ident: str, team: Team = None):
        if ident in self.teams:
            return
        if team is None:
            team = Team(ident)

        self.teams[ident] = team

    def update_data(self, inc):
        for t in ['H', 'A']:
            self.add_team(inc[t + 'ID'])
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

    def __str__(self):
        teams = list(self.teams.keys())
        teams_str = ', '.join(str(t) for t in teams)
        return 'League: {}, season: {}, teams {}: {}'.format(self.id, self.season,len(teams), teams_str)


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
            h_avg, a_avg = league.get_goal_avg(today_games['HID'].iloc[i], today_games['AID'].iloc[i])
            diff = h_avg - a_avg
            if diff > 0.5:  # win home
                odds = 0
            elif diff < -0.7:  # win away
                odds = 2
            else:
                odds = 1

            bets[i, odds] = min_bet
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
