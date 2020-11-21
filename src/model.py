import numpy as np
import pandas as pd
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
RETRAIN_MODELS = False
PI_GAME_LIMIT = 10
DIFF_TRESHOLD = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RES_COLS = ['H', 'D', 'A']
ODDS_COLS = ['OddsH', 'OddsD', 'OddsA']
SCORE_COLS = ['HSC', 'ASC']
TEAM_COLS = ['HID', 'AID']


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
        self.model: nn.Module = ProbPredictor()
        self.model_trained: bool = False
        self.train_data: List[List[float]] = []
        self.last_train_size = 0

    def add_team_if_missing(self, ident: str, team: Team = None):
        if ident in self.teams:
            return
        if team is None:
            team = Team(ident)

        self.teams[ident] = team

    def revise_pi_rating(self, hid, aid, hsc, asc, lr=0.1, gamma=0.3):
        exp_dif = self.predict_goal_diff(hid, aid)
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
        enough_games = self.teams[hid].n_games > PI_GAME_LIMIT and self.teams[aid].n_games > PI_GAME_LIMIT
        self.teams[hid].n_games += 1
        self.teams[aid].n_games += 1
        self.n_games += 1
        r = inc[RES_COLS].values
        winner = (r * np.array([1, 0, 2])).sum()
        if enough_games:
            self.train_data.append([exp_gd, winner])
        # if self.last_train_size + 2000 < len(self.train_data):
        #     print("TRAINING model for league {} on {} data points".format(self.id, len(self.train_data)))
        #     self.model = train_predictor(model=ProbPredictor(), data=self.train_data, plot=True)
        #     self.last_train_size = len(self.train_data)
        #     self.model_trained = True
        return exp_gd, winner, enough_games

    def predict_goal_diff(self, hid, aid):
        self.add_team_if_missing(hid)
        self.add_team_if_missing(aid)
        return self.teams[hid].expected_goal_diff(home=True) - self.teams[aid].expected_goal_diff(home=False)


    def __str__(self):
        teams = list(self.teams.keys())
        teams_str = ', '.join(str(t) for t in teams)
        return 'League: {}, games played: {}, training data {}'.format(self.id, self.n_games, len(self.train_data))


class Model:
    def __init__(self):
        self.bankroll: float = 0.0
        self.bankroll_variance = [float('inf'), float('-inf')]
        self.leagues: Dict[str, League] = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.model_win: nn.Module = ProbPredictor()
        self.model_lose: nn.Module = ProbPredictor()
        self.model_draw: nn.Module = ProbPredictor()
        self.train_data: List[List[float]] = []
        self.last_train_size = -100000
        self.diffs = []
        self.first = True
        self.opportunities: int = 0
        self.bets: int = 0

    def update_bankroll(self, curr):
        self.bankroll = curr
        self.bankroll_variance[0] = min(self.bankroll_variance[0], self.bankroll)
        self.bankroll_variance[1] = max(self.bankroll_variance[1], self.bankroll)

    def get_bet_size(self, min_bet, max_bet, br_norm):
        return max(min_bet, min(max_bet, self.bankroll / br_norm))

    def train_models(self):
        data = np.array(self.train_data)
        data_win = data.copy()
        data_win[data_win[:, 1] != 1, 1] = 0
        data_win[data_win[:, 1] == 1, 1] = 1
        data_lose = data.copy()
        data_lose[data_lose[:, 1] != 2, 1] = 0
        data_lose[data_lose[:, 1] == 2, 1] = 1
        data_draw = data.copy()
        data_draw[data_draw[:, 1] != 0, 1] = 2
        data_draw[data_draw[:, 1] == 0, 1] = 1
        data_draw[data_draw[:, 1] == 2, 1] = 0
        self.model_win = train_predictor(model=ProbPredictor(), data=data_win, plot=True)
        self.model_lose = train_predictor(model=ProbPredictor(), data=data_lose, plot=True)
        self.model_draw = train_predictor(model=ProbPredictor(), data=data_draw, plot=True)

    def place_bets(self, opps: pd.DataFrame, summary: pd.DataFrame, inc: pd.DataFrame):
        self.update_data(inc)
        if (self.first or RETRAIN_MODELS) and self.last_train_size + 10000 < len(self.train_data):
            print("TRAINING on {} data points".format(len(self.train_data)))
            self.train_models()
            self.last_train_size = len(self.train_data)
            self.first = False
        _summary = summary.iloc[0].to_dict()
        self.update_bankroll(_summary['Bankroll'])
        min_bet = _summary['Min_bet']
        max_bet = _summary['Max_bet']
        date = _summary['Date']
        today_games = opps[opps['Date'] == date]
        N = len(today_games)
        self.opportunities += N
        bets = np.zeros((N, 3))
        pred_outcome = np.zeros((1, 1))
        for i, game_row in enumerate(today_games.iterrows()):
            _, game = game_row
            if game['LID'] not in self.leagues:
                continue
            league = self.leagues[game['LID']]
            usable, pred_prob = predict_match_outcome(self.model_win, self.model_draw, self.model_lose, league, game['HID'], game['AID'])
            # if not usable:
            #     continue
            odds = game[ODDS_COLS].to_numpy()
            b_prob = odds2prob(odds)
            diff = pred_prob - b_prob
            dif_idx = np.argmax(diff)
            dif_max = np.max(diff)
            self.diffs.append(dif_max)
            bet = 0 if dif_max < DIFF_TRESHOLD else self.get_bet_size(min_bet, max_bet, 100)
            if bet > 0:
                self.bets += 1
            bets[i, dif_idx] = bet
            if VERBOSE and bet >= min_bet:
                p = [1, 0, 2]
                print('Odds: {}, Prediction: {}, Betting {} on {}'.format(b_prob, pred_prob, bet, p[dif_idx]))
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
        try:
            import matplotlib.pyplot as plt
            plt.hist(self.diffs)
            plt.show()
        except Exception:
            pass
        print(self.bankroll_variance)
        print(self.bets / self.opportunities)
        # for _, league in self.leagues.items():
        #     print(league)


def train_predictor(model, data, batch_size=32, lr=1e-4, m_epochs=50, plot=False) -> nn.Module:
    # _data = np.array(data, dtype=float)
    labels = np.array(data[:, 1], dtype=float)
    data = data[:, 0].reshape(-1, 1)
    dataset = ProbDataset(data, labels)
    train_len = int(len(dataset) * .7)
    train, test = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train, batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    epoch_loss = np.zeros(m_epochs)
    validation_loss = np.zeros(m_epochs)
    best_validation = float('inf')
    model.to(DEVICE)
    best_model = deepcopy(model)
    for epoch in range(m_epochs):
        avg_epoch = 0.
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
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
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                avg_eval += loss.item() / len(test_loader)
        validation_loss[epoch] = avg_eval
        # Model Checkpoint for best model
        if avg_eval < best_validation:
            best_validation = avg_eval
            best_model = deepcopy(model)

    if plot:
        # for name, param in best_model.named_parameters():
        #     print(name, param)
        try:
            import matplotlib.pyplot as plt
            plt.plot(epoch_loss)
            plt.plot(validation_loss)
            plt.legend(['train', 'validation'])
            plt.show()
        except Exception:
            pass
    best_model.eval()
    return best_model


class ProbDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index, :]), self.labels[index]


class ProbPredictor(nn.Module):
    def __init__(self, inp=1, h1=32):
        super(ProbPredictor, self).__init__()
        # torch.manual_seed(42)
        self.linear1 = nn.Linear(inp, h1)
        # nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.zeros_(self.linear1.bias)
        self.linear2 = nn.Linear(h1, 1)
        # self.linear3 = nn.Linear(h1, 3)
        # nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


def predict_match_outcome(m_win, m_draw, m_lose, league, hid, aid):
    pred_outcome = np.zeros((1, 1))
    pred_outcome[0, 0] = league.predict_goal_diff(hid, aid)
    enough_games = league.teams[hid].n_games > PI_GAME_LIMIT and league.teams[aid].n_games > PI_GAME_LIMIT
    pred_outcome_tensor = torch.tensor(pred_outcome, dtype=float).to(DEVICE)
    win = m_win(pred_outcome_tensor).cpu().detach().numpy()[0, 0]
    draw = m_draw(pred_outcome_tensor).cpu().detach().numpy()[0, 0]
    lose = m_lose(pred_outcome_tensor).cpu().detach().numpy()[0, 0]
    pred_prob = np.array([win, draw, lose])
    pred_prob = pred_prob / np.sum(pred_prob)
    return enough_games, pred_prob


if __name__ == "__main__":
    odd = np.array([3.1, 2.2, 1.1])
    prob = odds2prob(odd)
    print(prob)
