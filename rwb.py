import yfinance as yf
import itertools

class RWB:
    def __init__(self, one, two, three):
        self.ticker = one
        self.start = two
        self.today = three

    def analysis(self):
        # Activating Yahoo finance
        x = yf.Ticker(str(self.ticker))
        df = x.history(period='1d', start = self.start, end = self.today)
        
        # Moving Averages
        mas = [2, 3, 5, 8, 10, 20, 30, 40, 50, 60]
        for x in mas:
            df["MA"+str(x)] = round(df.iloc[:, 3].ewm(span=x, adjust=False).mean(), 2)

        pos = 0
        num = 0
        percentchange = []
        date = []
        # Conducting trades
        for i in df.index:
            cmin = min(df["MA2"][i], df["MA3"][i], df["MA5"][i], df["MA8"][i], df["MA10"][i])
            cmax = max(df["MA20"][i], df["MA30"][i], df["MA40"][i], df["MA50"][i], df["MA60"][i])
            close = df["Close"][i]
            if cmin > cmax:
                # RWB
                if pos == 0:
                    bp = close
                    pos = 1
                    # BUY
            elif cmin < cmax:
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

# x = RWB('AAPL', '2020-10-01', '2021-10-01')
# y, t, r = x.calculator()
# print(y)
# print(t)
# print(r)