from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
from ml.main import ml_api
from utils import get_db, exception_as_dict
import csv

app = Flask(__name__)
cors = CORS(app)

db_folder = Path("db/")
db_columns = ['Timestamp', 'Temperature', 'Load']

app.register_blueprint(ml_api, url_prefix='/ml')


@app.route('/')
def load_list():
    return jsonify("hello")


@app.route('/<int:device>')
@app.route('/<int:device>/latest')
@app.route('/<int:device>/<int:count>')
def load_detail_list_count(device, count=1):
    try:
        # with open(db_folder / (str(device) + ".csv"), 'r') as f:
        with open(get_db(device), 'r') as f:
            result = []
            i = 0
            for row in reversed(list(csv.reader(f))):
                result.append(dict(zip(db_columns, row)))
                i += 1
                if i == count:
                    return jsonify(result)
    except Exception as e:
        return jsonify(exception_as_dict(e))


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

        # now = datetime.now()
        # midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # day = now.day
        # month = now.month
        # year = now.year
        # seconds = (now - midnight).seconds

        time = int(datetime.now().timestamp())

        row = [time, temp, load]
        with open(get_db(device), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return jsonify(True)
    except Exception as e:
        return jsonify(exception_as_dict(e))


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0")
