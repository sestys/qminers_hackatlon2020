import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from model import Model
from environment import Environment
RUNS = 10
bankrolls = np.zeros((RUNS,), dtype=np.float)
for run in range(RUNS):
    dataset = pd.read_csv('../data/training_data.csv', parse_dates=['Date', 'Open'])
    model = Model()
    env = Environment(dataset, model, init_bankroll=1000., min_bet=5., max_bet=100.)
    # evaluation = env.run(start=pd.to_datetime('2005-07-01'), end=pd.to_datetime('2006-06-30'))
    evaluation = env.run(start=pd.to_datetime('2005-07-01'))

    print('Run {}: final bankroll: {}'.format(run + 1, env.bankroll))
    bankrolls[run] = env.bankroll

    # print(f'Final bankroll: {env.bankroll:.2f}')

mean = bankrolls.mean()
std = bankrolls.std()
mi = bankrolls.min()
ma = bankrolls.max()
print('Mean: {}, STD: {}, MIN: {}, MAX: {}'.format(mean, std, mi, ma))
plt.plot(bankrolls)
plt.legend('Bankroll over runs')
plt.show()
