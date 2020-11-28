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
RETRAIN_MODELS = False
PI_GAME_LIMIT = 10
DIFF_TRESHOLD = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RES_COLS = ['H', 'D', 'A']
ODDS_COLS = ['OddsH', 'OddsD', 'OddsA']
SCORE_COLS = ['HSC', 'ASC']
TEAM_COLS = ['HID', 'AID']
MODEL_COL = ['H_HRTG', 'H_ARTG', 'A_HRTG', 'A_ARTG', 'EGD']
LEAGUE_DATA = ['Sea'] + TEAM_COLS + SCORE_COLS + RES_COLS + MODEL_COL + ['UFT']
LEAGUE_DATA2 = ['Sea'] + TEAM_COLS + SCORE_COLS + RES_COLS

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
    def __init__(self, in_size):
        super(ProbPredictor, self).__init__()
        self.layer_1 = nn.Linear(in_size, 32)
        self.layer_2 = nn.Linear(32, 3)
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
        self.data = pd.DataFrame(columns=LEAGUE_DATA)
        self.data.Sea.astype('datetime64')
        self.data.UFT.astype(bool)

    def add_team_if_missing(self, ident: str, team: Team = None):
        if ident in self.teams:
            return
        if team is None:
            team = Team(ident)

        self.teams[ident] = team

    def revise_pi_rating(self, hid, aid, hsc, asc, lr=0.1, gamma=0.3):
        _, exp_dif = self.predict_match_outcome(hid, aid)
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
        if self.season != inc['Sea']:
            # print('New Season league {}'.format(self.id))
            # home_mean = self.data.HSC[self.data.Sea == self.season].mean()
            # home_std = self.data.HSC[self.data.Sea == self.season].std()
            # away_mean = self.data.ASC[self.data.Sea == self.season].mean()
            # away_std = self.data.ASC[self.data.Sea == self.season].std()
            # home_win_pct = self.data.H[self.data.Sea == self.season].mean()
            # away_win_pct = self.data.A[self.data.Sea == self.season].mean()
            # draw_pct = self.data.D[self.data.Sea == self.season].mean()
            # if self.season != 0:
            #     print(home_mean, home_std, away_mean, away_std, home_win_pct, away_win_pct, draw_pct)

            self.season = inc['Sea']

        hid = inc['HID']
        aid = inc['AID']
        self.add_team_if_missing(hid)
        self.add_team_if_missing(aid)
        self.data = self.data.append(inc[LEAGUE_DATA2])
        self.data['H_HRTG'].iloc[-1] = self.teams[hid].R_home
        self.data['H_ARTG'].iloc[-1] = self.teams[hid].R_away
        self.data['A_HRTG'].iloc[-1] = self.teams[aid].R_home
        self.data['A_ARTG'].iloc[-1] = self.teams[aid].R_away
        exp_gd = self.revise_pi_rating(hid, aid, inc['HSC'], inc['ASC'])
        enough_games = self.teams[hid].n_games > PI_GAME_LIMIT and self.teams[aid].n_games > PI_GAME_LIMIT
        self.teams[hid].n_games += 1
        self.teams[aid].n_games += 1
        self.n_games += 1
        r = inc[RES_COLS].values
        winner = (r * np.array([1, 0, 2])).sum()
        self.data['EGD'].iloc[-1] = exp_gd
        self.data['UFT'].iloc[-1] = enough_games

        return exp_gd, winner, enough_games


    def predict_match_outcome(self, hid, aid):
        self.add_team_if_missing(hid)
        self.add_team_if_missing(aid)
        h_team = self.teams[hid]
        a_team = self.teams[aid]
        enough_games = self.teams[hid].n_games > PI_GAME_LIMIT and self.teams[aid].n_games > PI_GAME_LIMIT
        return enough_games, h_team.expected_goal_diff(home=True) - a_team.expected_goal_diff(home=False)

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
        self.model: nn.Module = ProbPredictor(len(MODEL_COL))
        self.train_data: List[List[float]] = []
        self.train = True
        self.diffs = []
        self.placed_bets = [0]
        self.bet_size = []
        self.counter = 0

    def train_predictor(self, data, labels, batch_size=32, lr=1e-4, m_epochs=20, plot=False) -> None:
        dataset = ProbDataset(data, labels)
        train_len = int(len(dataset) * .7)
        train, test = random_split(dataset, [train_len, len(dataset) - train_len])
        train_loader = DataLoader(train, batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size, shuffle=True)
        model = ProbPredictor(data.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        epoch_loss = np.zeros(m_epochs)
        validation_loss = np.zeros(m_epochs)
        best_validation = float('inf')
        model.to(DEVICE)
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
                self.model = deepcopy(model)

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


    def update_bankroll(self, curr):
        self.bankroll.append(curr)
        self.bankroll_variance[0] = min(self.bankroll_variance[0], curr)
        self.bankroll_variance[1] = max(self.bankroll_variance[1], curr)

    def predict_model(self, goal_diff, home_team, away_team):
        inp = np.zeros((1, len(MODEL_COL)))
        inp[0, 0] = home_team.R_home
        inp[0, 1] = home_team.R_away
        inp[0, 2] = away_team.R_home
        inp[0, 3] = away_team.R_away
        inp[0, 4] = goal_diff
        inp = torch.tensor(inp, dtype=float)
        inp.to(DEVICE)
        pred_prob = F.softmax(self.model(inp), dim=1).cpu().detach().numpy()
        return pred_prob[0, [1, 0, 2]]

    def compute_bet(self, p_prob, b_prob, min_bet, max_bet, bet):
        diff = p_prob - b_prob
        dif_idx = np.argmax(diff)
        dif_max = np.max(diff)
        self.diffs.append(dif_max)
        if dif_max < 0.05:
            bet = 0
        return bet, dif_idx

    def _train(self):
        league_data = []
        for _, league in self.leagues.items():
            league_data.append(league.data.loc[league.data['UFT'] == True])

        df = pd.concat(league_data)
        data = df[MODEL_COL].to_numpy().astype(float)
        label = (df[RES_COLS].to_numpy() * np.array([1, 0, 2])).sum(axis=1).astype('int64')
        self.train_predictor(data=data, labels=label, plot=True)
        self.model.eval()
        self.train = False

    def place_bets(self, opps: pd.DataFrame, summary: pd.DataFrame, inc: pd.DataFrame):
        self.update_data(inc)
        self.counter += 1
        if self.train or self.counter == 7000:
            self._train()
            print('TRAINING')
            # data = np.array(self.train_data, dtype=float)
            # labels = np.array(data[:, 1], dtype=int)
            # data = data[:, 0].reshape(-1, 1)
            # self.train_predictor(data=data, labels=labels, plot=False)
            # self.model.eval()
            # self.train = False
        _summary = summary.iloc[0].to_dict()
        self.update_bankroll(_summary['Bankroll'])
        min_bet = _summary['Min_bet']
        max_bet = _summary['Max_bet']
        date = _summary['Date']
        today_games = opps[opps['Date'] == date]
        N = len(today_games)
        bets = np.zeros((N, 3))
        placed_bets = 0
        bet = max(min_bet, min(max_bet, self.bankroll[-1] / 100))
        for i in range(N):
            game = today_games.iloc[i]
            if game['LID'] not in self.leagues:
                continue
            league = self.leagues[game['LID']]
            use, goal_dif = league.predict_match_outcome(game['HID'], game['AID'])
            if not use:
                continue
            p_prob = self.predict_model(goal_dif, league.teams[game['HID']], league.teams[game['AID']])
            b_prob = odds2prob(game[self.odds_cols].to_numpy())

            bet, idx = self.compute_bet(p_prob, b_prob, min_bet, max_bet, bet)
            if bet >= min_bet:
                placed_bets += 1
            bets[i, idx] = bet

        self.bet_size.append(bet)
        self.placed_bets.append(placed_bets + self.placed_bets[-1])
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
        plot(self.bankroll)
        plot(self.placed_bets)
        plot(self.bet_size)


def plot(data, title=''):
    try:
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.show()
    except Exception:
        pass