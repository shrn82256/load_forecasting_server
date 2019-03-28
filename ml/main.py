from flask import Blueprint, jsonify
from . import train, execute, test
import os

ml_api = Blueprint('ml_api', __name__)


@ml_api.route("/ml/train/<int:device>")
def trainModel(device):
    db = "db/{}.csv".format(device)
    if os.path.isfile(db):
        train.main(device, db, model_architecture="single_lstm")
        return "true"
    else:
        return jsonify(False)


@ml_api.route("/ml/exec/<int:device>")
def executeModel(device):
    db = "db/{}.csv".format(device)
    if os.path.isfile(db):
        return jsonify(execute.main(device, db))
    else:
        return jsonify(False)


@ml_api.route("/ml/test/<int:device>")
def testModel(device):
    db = "db/{}.csv".format(device)
    if os.path.isfile(db):
        return jsonify(test.main(device, db))
    else:
        return jsonify(False)
