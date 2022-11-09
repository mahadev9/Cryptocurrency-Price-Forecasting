
from distutils.command.clean import clean
import os
import sys
import glob
import shutil
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from datetime import *
from pathlib import Path

BASE_URL = 'https://data.binance.vision/'


OPEN_INDEX = 1
HIGH_INDEX = 2
LOW_INDEX = 3
CLOSE_INDEX = 4
VOLUME_INDEX = 5
IGNORE_INDEX = 11

BTC = 'BTCUSDT'
ETH = 'ETHUSDT'
MATIC = 'MATICUSDT'
BNB = 'BNBUSDT'
ADA = 'ADAUSDT'
DOGE = 'DOGEUSDT'
SYMBOLS = [BTC, ETH, MATIC, BNB, ADA, DOGE]


def get_download_url(file_url):
    return '{}{}'.format(BASE_URL, file_url)


def get_destination_dir(file_url, folder=None):
    store_directory = os.environ.get('STORE_DIRECTORY')
    if folder:
        store_directory = folder
    if not store_directory:
        store_directory = os.path.dirname(os.path.realpath(__file__))
    symbol = file_url.split('/')[-3]
    return os.path.join(store_directory, symbol)


def convert_to_date_object(d):
    year, month, day = [int(x) for x in d.split('-')]
    date_obj = date(year, month, day)
    return date_obj


def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
    trading_type_path = 'data/spot'
    if trading_type != 'spot':
        trading_type_path = f'data/futures/{trading_type}'
    if interval is not None:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
    else:
        path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
    return path


def download_file(base_path, file_name, date_range=None, folder=None):
    download_path = "{}{}".format(base_path, file_name)
    if folder:
        base_path = os.path.join(folder, base_path)
    if date_range:
        date_range = date_range.replace(" ", "_")
        base_path = os.path.join(base_path, date_range)
    save_folder = get_destination_dir(base_path, folder)
    save_path = os.path.join(save_folder, file_name)

    if os.path.exists(save_path):
        print("\nfile already exists! {}".format(save_path))
        return

    # make the directory
    if not os.path.exists(save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)

    try:
        download_url = get_download_url(download_path)
        dl_file = urllib.request.urlopen(download_url)
        length = dl_file.getheader('content-length')
        if length:
            length = int(length)
            blocksize = max(4096, length//100)

        with open(save_path, 'wb') as out_file:
            dl_progress = 0
            print("\nFile Download: {}".format(save_path))
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                dl_progress += len(buf)
                out_file.write(buf)
                done = int(50 * dl_progress / length)
                sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)))
                sys.stdout.flush()

        with zipfile.ZipFile(save_path, 'r') as zip_file:
            path = save_path.split(os.sep)[-3:-1]
            zip_file.extractall(os.path.join(*path))

    except urllib.error.HTTPError:
        print("\nFile not found: {}".format(download_url))
        pass


def download_monthly_klines(trading_type, symbols, intervals, start_date, end_date, folder):
    current = 0
    date_range = None
    num_symbols = len(symbols)
    folder = os.path.join(os.getcwd(), folder)

    if start_date and end_date:
        date_range = start_date + "_" + end_date

    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    years = [i for i in range(int(start_year), int(end_year)+1)]
    months = range(1, 12+1)

    start_date = convert_to_date_object(start_date)
    end_date = convert_to_date_object(end_date)

    print('Found {} symbols'.format(num_symbols))

    for symbol in symbols:
        print('[{}/{}] - start download monthly {} klines '.format(current +
              1, num_symbols, symbol))
        for interval in intervals:
            for year in years:
                for month in months:
                    current_date = convert_to_date_object(
                        '{}-{}-01'.format(year, month))
                    if current_date >= start_date and current_date <= end_date:
                        path = get_path(trading_type, 'klines',
                                        'monthly', symbol, interval)
                        file_name = '{}-{}-{}-{}.zip'.format(
                            symbol.upper(), interval, year, '{:02d}'.format(month))
                        download_file(path, file_name, date_range, folder)

        current += 1


def clean_data(folder):
    shutil.rmtree(os.path.join(os.getcwd(), folder))


def get_data(folder, symbol):
    path = os.path.join(folder, symbol, '*.csv')
    data_files = glob.glob(path)
    data = []
    for file in data_files:
        df = pd.read_csv(file, header=None)
        data.append(df)
    return pd.concat(data, axis=0, ignore_index=True)


def split_data(df, lookback, train_split_ratio, columns_to_keep):
    df = df[df[IGNORE_INDEX] == 0]
    df = df[columns_to_keep]
    data_raw = df.to_numpy()
    data = []

    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    train_set_size = int(np.round(train_split_ratio*data.shape[0]))

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, CLOSE_INDEX-1]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, CLOSE_INDEX-1]

    return [x_train, y_train, x_test, y_test]


if __name__ == '__main__':
    download_monthly_klines('spot', ['BTCUSDT'], ['15m'],
                            '2019-01-01', '2019-07-31', os.path.join(os.getcwd(), 'train'))
    # clean_data('train')
