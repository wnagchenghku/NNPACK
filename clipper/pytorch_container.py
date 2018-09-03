from __future__ import print_function
# import rpc
import os
import sys
import json

# import numpy as np
import cloudpickle
import torch
import importlib
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import argparse
import datetime

IMPORT_ERROR_RETURN_CODE = 3

PYTORCH_WEIGHTS_RELATIVE_PATH = "pytorch_weights.pkl"
PYTORCH_MODEL_RELATIVE_PATH = "pytorch_model.pkl"


def load_predict_func(file_path):
    if sys.version_info < (3, 0):
        with open(file_path, 'r') as serialized_func_file:
            return cloudpickle.load(serialized_func_file)
    else:
        with open(file_path, 'rb') as serialized_func_file:
            return cloudpickle.load(serialized_func_file)


def load_pytorch_model(model_path, weights_path):
    if sys.version_info < (3, 0):
        with open(model_path, 'r') as serialized_model_file:
            model = cloudpickle.load(serialized_model_file)
    else:
        with open(model_path, 'rb') as serialized_model_file:
            model = cloudpickle.load(serialized_model_file)

    model.load_state_dict(torch.load(weights_path))
    return model


# class PyTorchContainer(rpc.ModelContainerBase):
class PyTorchContainer():
    # def __init__(self, path, input_type):
    def __init__(self, model_name):
        # self.input_type = rpc.string_to_input_type(input_type)
        # modules_folder_path = "{dir}/modules/".format(dir=path)
        modules_folder_path = "model/{model_name}".format(model_name=model_name)
        sys.path.append(os.path.abspath(modules_folder_path))
        predict_fname = "func.pkl"
        # predict_path = "{dir}/{predict_fname}".format(
        #     dir=path, predict_fname=predict_fname)
        predict_path = "{dir}/{predict_fname}".format(
            dir=modules_folder_path, predict_fname=predict_fname)

        self.predict_func = load_predict_func(predict_path)

        # torch_model_path = os.path.join(path, PYTORCH_MODEL_RELATIVE_PATH)
        # torch_weights_path = os.path.join(path, PYTORCH_WEIGHTS_RELATIVE_PATH)
        torch_model_path = os.path.join(modules_folder_path, PYTORCH_MODEL_RELATIVE_PATH)
        torch_weights_path = os.path.join(modules_folder_path, PYTORCH_WEIGHTS_RELATIVE_PATH)

        self.model = load_pytorch_model(torch_model_path, torch_weights_path)

    # def predict_ints(self, inputs):
    #     preds = self.predict_func(self.model, inputs)
    #     return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = self.predict_func(self.model, inputs)
        # return [str(p) for p in preds]

    # def predict_doubles(self, inputs):
    #     preds = self.predict_func(self.model, inputs)
    #     return [str(p) for p in preds]

    # def predict_bytes(self, inputs):
    #     preds = self.predict_func(self.model, inputs)
    #     return [str(p) for p in preds]

    # def predict_strings(self, inputs):
    #     preds = self.predict_func(self.model, inputs)
    #     return [str(p) for p in preds]


if __name__ == "__main__":
    print("Starting PyTorchContainer container")
    # rpc_service = rpc.RPCService()
    parser = argparse.ArgumentParser(description='Parse model name.')
    parser.add_argument('-m', type=str, dest="model_name", action="store", required=True)
    args = parser.parse_args()

    f = open('/tmp/tmp.txt', 'w')

    load_start = datetime.datetime.now()
    try:
        # model = PyTorchContainer(rpc_service.get_model_path(),
        #                          rpc_service.get_input_type())
        model = PyTorchContainer(args.model_name)
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)
    load_end = datetime.datetime.now()

    # rpc_service.start(model)
    inputs = torch.randn(1, 3, 224, 224)

    predict_start = datetime.datetime.now()
    model.predict_floats(torch.autograd.Variable(inputs))
    predict_end = datetime.datetime.now()

    f.write("model loading takes %d s, %d us\n" % ((load_end - load_start).seconds, (load_end - load_start).microseconds))
    f.write("prediction takes %d s, %d us\n" % ((predict_end - predict_start).seconds, (predict_end - predict_start).microseconds))
    f.close()
