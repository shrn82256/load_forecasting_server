from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
from ml.main import ml_api
import pandas as pd
import numpy as np
import json
import csv
import os

app = Flask(__name__)
cors = CORS(app)

db_folder = Path("db/")
db_columns = ['Day', 'Month', 'Year', 'Seconds', 'Temperature', 'Load']

app.register_blueprint(ml_api)


@app.route('/')
def load_list():
    return jsonify("hello")


""" def load_detail_list(device):
    df = pd.read_csv(db_folder / (str(device) + ".csv"),
                     names=db_columns)
    return df.to_json(orient='records') """


@app.route('/<int:device>/<int:count>')
def load_detail_list_count(device, count):
    with open(db_folder / (str(device) + ".csv"), 'r') as f:
        result = []
        i = 0
        for row in reversed(list(csv.reader(f))):
            result.append(dict(zip(db_columns, row)))
            i += 1
            if i == count:
                return jsonify(result)


@app.route('/<int:device>')
@app.route('/<int:device>/latest')
def load_detail_list_latest(device):
    with open(db_folder / (str(device) + ".csv"), 'r') as f:
        for row in reversed(list(csv.reader(f))):
            return jsonify(dict(zip(db_columns, row)))


@app.route('/log', methods=['GET'])
def load_entry():
    try:
        q = ""
        for i in str(request.args.get('q')):
            # q += chr(ord(i)-5)
            q += i

        device = str(int(q[:-10]))
        temp = str(int(q[-10:-7]))
        load = str(int(q[-7:]))

        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        day = now.day
        month = now.month
        year = now.year
        seconds = (now - midnight).seconds

        row = [day, month, year, seconds, temp, load]
        with open(db_folder / (device + ".csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return jsonify(True)

    except Exception as e:
        print(e)
        return jsonify(False)


"""
@app.route('/myth.html')
def load_detail_list():
    return render_template('myth.html', the_title='Tiger in Myth and Legend')
 """

if __name__ == '__main__':
    app.run(debug=True)
