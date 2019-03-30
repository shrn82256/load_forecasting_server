from flask import Blueprint, jsonify
from . import train, execute, test
from utils import get_db, exception_as_dict
import os

ml_api = Blueprint('ml_api', __name__)
# model_architecture="triple_lstm"
# model_architecture="single_lstm"
model_architecture = "double_lstm"


@ml_api.route("/ml/train/<string:device>")
def train_model(device):
    try:
        db = get_db(device)
        if os.path.isfile(db):
            return jsonify(train.main(device, db, model_architecture=model_architecture))
        else:
            return jsonify(False)
    except Exception as e:
        return jsonify(exception_as_dict(e))


@ml_api.route("/ml/exec/<string:device>")
def execute_model(device):
    try:
        db = get_db(device)
        if os.path.isfile(db):
            return jsonify({"next": execute.main(device, db, model_architecture=model_architecture)})
        else:
            return jsonify(False)
    except Exception as e:
        return jsonify(exception_as_dict(e))


@ml_api.route("/ml/test/<string:device>")
def test_model(device):
    try:
        db = get_db(device)
        if os.path.isfile(db):
            return jsonify(
                test.main(device, db, model_architecture=model_architecture))
        else:
            return jsonify(False)
    except Exception as e:
        return jsonify(exception_as_dict(e))
