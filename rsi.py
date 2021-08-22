import yfinance as yf
import pandas as pd
import numpy as np
import itertools

class RSI:
    def __init__(self, one, two, three):
        self.ticker = one
        self.start = two
        self.today = three

    def calculator(self):
        # Activating Yahoo finance
        x = yf.Ticker(str(self.ticker))
        df = x.history(period='1d', start = self.start, end = self.today)

        # Get the difference in price from previous price
        series = df['Close'].copy().diff()

        # Get upwards and downwards gains
        up, down = series.copy(), series.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the exponential weighted values
        roll_up1 = up.ewm(span=14, min_periods=0, adjust=False, ignore_na=False).mean()
        roll_down1 = down.abs().ewm(span=14, min_periods=0, adjust=False, ignore_na=False).mean()

        # Calculate the RSI based on EMA
        rs1 = roll_up1 / roll_down1
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs1))

        # Calculate WMA
        weights = np.array([0.5, 0.25, 0.25])
        sum_weights = np.sum(weights)
        df['weighted_ma'] = (df['Close']
                             .rolling(window=3, center=True)
                             .apply(lambda y: np.sum(weights * y) / sum_weights, raw=False))
        df = pd.DataFrame(df)
        pos = 0
        num = 0
        d = 0
        percentchange = []
        date = []
        for i in df.index:
            wma = df['weighted_ma'][i]
            close = df["Close"][i]
            rsi = df['RSI'][i]
            if wma < close and rsi < 50:
                if pos == 0:
                    bp = close
                    pos = 1
                    # BUY
            elif wma > close and rsi > 50:
                # BWR
                if pos == 1:
                    pos = 0
                    sp = close
                    # SELL
                    pc = (sp / bp - 1) * 100
                    percentchange.append(float(pc))
                    date.append(i)

            if num == df["Close"].count() - 1 and pos == 1:
                pos = 0
                sp = close
                # SELL
                pc = (sp / bp - 1) * 100
                percentchange.append(float(pc))
                date.append(i)

            num += 1

        gains = 0
        ng = 0
        losses = 0
        nl = 0
        total_return = 1
        d += 1

        for i in percentchange:
            if i > 0:
                gains += i
                ng += 1
            else:
                losses += i
                nl += 1
            total_return = total_return * ((i / 100) + 1)

        total_return = round((total_return - 1) * 100, 2)

        if ng > 0:
            avg_gain = round((gains / ng), 2)
            max_return = (round(max(percentchange), 2))
        else:
            avg_gain = 0
            max_return = "undefined"

        if nl > 0:
            avg_loss = round((losses / nl), 2)
            max_loss = (round(min(percentchange), 2))
            ratio = (round((-avg_gain / avg_loss), 2))
        else:
            avg_loss = 0
            max_loss = "undefined"
            ratio = "inf"

        # Results
        x = [ng + nl, ratio, avg_gain, avg_loss, max_return, max_loss,
             total_return]

        returns = list(itertools.repeat(float(1), len(df.index)))
        for i in range(len(df.index)):
            for j in range(len(date)):
                if df.index[i] == date[j]:
                    returns[i] = (percentchange[j]/100)+1
        for i in range(len(df.index)):
            if i == df.index[0]:
                returns[0] = 1
            else:
                temp = returns[i-1]
                returns[i] = temp * returns[i]

        return x, returns, list(df.index)

# x = RSI('AAPL', '2020-10-01', '2021-10-01')
# y, t, r = x.calculator()
# print(y)
# print(t)
# print(r)