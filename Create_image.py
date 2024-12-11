# Import Libraries
# -*- coding: utf-8 -*-
# 서버 설치에 모듈 설치 시 --user 이용
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.patches as patches
import os

# Parameters
root_dir = '/home/juwonkim/JPmorgon_CNN_repo/NLP_project'
window_size = 60

def plotgraph(start, finish, dir_plot, formatted_date, df):
    periods = finish - start

    # Create figure and main Axes for candlestick chart using plt.figure()
    fig = plt.figure(figsize=(6, 4), dpi=60)
    ax_candle = fig.add_subplot(111)

    data_slice = df.iloc[start:finish]
    open = data_slice['open'].tolist()
    high = data_slice['high'].tolist()
    low = data_slice['low'].tolist()
    close = data_slice['close'].tolist()
    volume = data_slice['volume'].tolist()

    # 날짜 데이터를 Matplotlib의 날짜 포맷으로 변환
    dates = mdates.date2num(pd.date_range(start=formatted_date, periods=periods, freq='D'))

    for i, (date_num, op, cl, hi, lo) in enumerate(zip(dates, open, close, high, low)):
        width = 0.25 + i * 0.02
        color = 'g' if cl >= op else 'r'
        lower = min(op, cl)
        height = abs(cl - op)
        ax_candle.plot([date_num, date_num], [lo, hi], color='k', linewidth=1, zorder=1)
        rect = patches.Rectangle((date_num - width / 2, lower), width, height, color=color, zorder=2)
        ax_candle.add_patch(rect)

    ax_volume = ax_candle.twinx()
    ax_volume.bar(dates, volume, color='blue', alpha=0.3, width=0.8, align='center')
    ax_volume.set_ylim(0, max(volume) * 3)
    #ax_volume.autoscale_view()

    # Apply auto-scaling and turn off all axes
    ax_candle.autoscale_view()
    ax_candle.axis('off')
    ax_volume.axis('off')

    
    filename = os.path.join(dir_plot, f"{formatted_date}.jpg")
    
    #plt.autoscale()
    #plt.axis('off')
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)




# List of stocks that comprise the s&p 500 <-- Download from Wikipidia
tickers = pd.read_csv('/home/juwonkim/JPmorgon_CNN_repo/NLP_project/constituents.csv')['Symbol'].tolist()
total_num = 0
for i, ticker in enumerate(tickers):
    print(str(i) + ' 번째 ', ticker)
    
    csv_path = os.path.join(root_dir, "data", "raw", ticker + '.csv')
    save_dirs = [
        os.path.join(root_dir, "data", "csv", ticker),
        os.path.join(root_dir, "data", "plot", ticker)
    ]
    for dir in save_dirs:
        os.makedirs(dir, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert date column to datetimes
    df.sort_values('Date', inplace=True)  # Sort data by date
    df = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= '2015-12-31')]  # Filter data

    df = df[['Date','Open','High','Low','Close','Volume']]
    df = df.rename({'Date':'date','High':'high','Low':'low','Open':'open','Close':'close', 'Volume':'volume'}, axis=1)
    df['date'] = pd.to_datetime(df['date'])
    # Set the directories

    # Loop for all the approaches
    plot_dir = root_dir + "/data/plot/" + ticker + "/"
    
    for i, start_date in enumerate(df['date'][:-window_size]):
        formatted_date = start_date.strftime('%Y-%m-%d')
        plotgraph(i, i + window_size, plot_dir, formatted_date, df)

    total_num+=1
    print('다운받은 주식 수: ', total_num)
