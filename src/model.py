import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from math import log
from typing import List, Dict
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

VERBOSE = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

res_cols = ['H', 'D', 'A']


class ProbDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        d = torch.tensor(self.data[index, :])
        l = self.labels[index]
        return d, l


class ProbPredictor(nn.Module):
    def __init__(self):
        super(ProbPredictor, self).__init__()
        self.layer_1 = nn.Linear(1, 10)
        self.layer_2 = nn.Linear(10, 3)
        # self.softmax = F.log_softmax(nclass)

    def forward(self, x):
        x = x.float()
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x


def psi(e: float, b: int = 10, c: int = 3) -> float:
    return c * log(1 + e, b)


def odds2prob(odds):
    o = 1 / odds
    norm = o.sum()
    return o / norm


class Team:
    def __init__(self, ident: str):
        self.id: str = ident
        self.season: int = 0
        self.n_games: int = 0
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
        return exp_dif


    def update_data(self, inc):
        """
        :param inc:
        :return: pregame expected goal difference and the outcome of the game
        """
        hid = inc['HID']
        aid = inc['AID']
        exp_gd = self.revise_pi_rating(hid, aid, inc['HSC'], inc['ASC'])
        enough_games = self.teams[hid].n_games > 10 and self.teams[aid].n_games > 10
        self.teams[hid].n_games += 1
        self.teams[aid].n_games += 1
        self.n_games += 1
        r = inc[res_cols].values
        winner = (r * np.array([1, 0, 2])).sum()
        return exp_gd, winner, enough_games


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
        self.bankroll: List[float] = []
        self.bankroll_variance = [float('inf'), float('-inf')]
        self.leagues: Dict[str, League] = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.odds_cols = ['OddsH', 'OddsD', 'OddsA']
        self.score_cols = ['HSC', 'ASC']
        self.team_cols = ['HID', 'AID']
        self.res_cols = ['H', 'D', 'A']
        self.model: nn.Module = ProbPredictor()
        self.train_data: List[List[float]] = []
        self.train = True
        self.diffs = []

    def train_predictor(self, data, labels, batch_size=32, lr=1e-3, m_epochs=10, plot=False) -> None:
        dataset = ProbDataset(data, labels)
        train_len = int(len(dataset) * .7)
        train, test = random_split(dataset, [train_len, len(dataset) - train_len])
        train_loader = DataLoader(train, batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size, shuffle=True)
        model = ProbPredictor()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        epoch_loss = np.zeros(m_epochs)
        validation_loss = np.zeros(m_epochs)
        best_validation = float('inf')
        model.to(device)
        for epoch in range(m_epochs):
            avg_epoch = 0.
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, y)
                avg_epoch += loss.item() / len(train_loader)
                loss.backward()
                optimizer.step()

            epoch_loss[epoch] = avg_epoch

            # Validation loop
            model.eval()
            avg_eval = 0.
            for x, y in test_loader:
                with torch.no_grad():
                    x = x.to(device)
                    y = y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    avg_eval += loss.item() / len(test_loader)
            validation_loss[epoch] = avg_eval
            # Model Checkpoint for best model
            if avg_eval < best_validation:
                best_validation = avg_eval
                self.model = deepcopy(model)

        # if plot:
        #     plt.plot(epoch_loss)
        #     plt.plot(validation_loss)
        #     plt.legend(['train', 'validation'])
        #     plt.show()

    def update_bankroll(self, curr):
        self.bankroll.append(curr)
        self.bankroll_variance[0] = min(self.bankroll_variance[0], curr)
        self.bankroll_variance[1] = max(self.bankroll_variance[1], curr)

    def predict_model(self, goal_diff):
        inp = np.zeros((1, 1))
        inp[0, 0] = goal_diff
        inp = torch.tensor(inp, dtype=float)
        inp.to(device)
        pred_prob = F.softmax(self.model(inp), dim=1).cpu().detach().numpy()
        return pred_prob[0, [1, 0, 2]]

    def compute_bet(self, p_prob, b_prob, min_bet, max_bet):
        diff = p_prob - b_prob
        dif_idx = np.argmax(diff)
        dif_max = np.max(diff)
        self.diffs.append(dif_max)
        if dif_max < 0.05:
            bet = 0
        else:
            bet = max(min_bet, min(max_bet, self.bankroll[-1] / 100))
        return bet, dif_idx

    def place_bets(self, opps: pd.DataFrame, summary: pd.DataFrame, inc: pd.DataFrame):
        self.update_data(inc)
        if self.train:
            data = np.array(self.train_data, dtype=float)
            labels = np.array(data[:, 1], dtype=int)
            data = data[:, 0].reshape(-1, 1)
            self.train_predictor(data=data, labels=labels, plot=False)
            self.model.eval()
            self.train = False
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
            p_prob = self.predict_model(league.predict_match_outcome(game['HID'], game['AID']))
            b_prob = odds2prob(game[self.odds_cols].to_numpy())

            bet, idx = self.compute_bet(p_prob, b_prob, min_bet, max_bet)
            bets[i, idx] = bet

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

            gd, winner, use = league.update_data(row)
            if use:
                self.train_data.append([gd, winner])

    def print_leagues(self):
        print(self.bankroll_variance)
        try:
            import matplotlib.pyplot as plt
            plt.plot(self.bankroll)
            plt.show()
            plt.hist(self.diffs)
            plt.show()
        except Exception:
            pass

