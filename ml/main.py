from flask import Blueprint, jsonify
from . import train, execute, test
import os

ml_api = Blueprint('ml_api', __name__)


@ml_api.route("/ml/train/<string:device>")
def train_model(device):
    db = "db/{}.csv".format(device)
    if os.path.isfile(db):
        train.main(device, db)
        return "true"
    else:
        return jsonify(False)


@ml_api.route("/ml/exec/<string:device>")
def execute_model(device):
    db = "db/{}.csv".format(device)
    if os.path.isfile(db):
        return jsonify({"next": execute.main(device, db)})
    else:
        return jsonify(False)


@ml_api.route("/ml/test/<string:device>")
def test_model(device):
    db = "db/{}.csv".format(device)
    if os.path.isfile(db):
        return jsonify(test.main(device, db))
    else:
        return jsonify(False)
