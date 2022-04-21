# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import copy
import yaml
import poptorch
import popdist.poptorch
import import_helper
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from pydantic import BaseModel
from cus_base64 import Base64Bytes
from inference import inference
import models
import datasets
import utils

class responseJSON(BaseModel):
    candidate: int   


class requestJSON(BaseModel):
    binary_image: Base64Bytes
    seed: int = 0

app = FastAPI()
with open('config.yml') as f:
    config = yaml.safe_load(f)

utils.Logger.setup_logging_folder(config)
@app.exception_handler(Exception)
async def unicorn_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__,
                 "message": exc.args[0]},
    )

if config['use_popdist']:
    opts = popdist.poptorch.Options(ipus_per_replica=len(config['pipeline_splits']) + 1)
else:
    opts = poptorch.Options()
    opts.replicationFactor(config['replicas'])
opts.deviceIterations(config['device_iterations'])

model = models.get_model(config, datasets.datasets_info[config['data']], pretrained=not config['random_weights'])
model.eval()
opts = utils.inference_settings(config, copy.deepcopy(opts))

file_name = "poptorch.model"
# input = torch.randn(1, 3, 244, 244)
# if not os.path.isfile(file_name):
#     inference_model = poptorch.inferenceModel(model, opts)
#     inference_model.compileAndExport(file_name, input)
inference_model = poptorch.load(file_name)
@app.post("/generate", response_model=responseJSON)
def generate(input_data: requestJSON):
    try:
        results = inference(
            inference_model,
            input_data
        )
        return results

    except Exception as exc:
        raise RuntimeError(exc)


@app.get("/")
def api_info():
    return {
        "server_configs": 'resnet',
        "request_json": requestJSON.schema(),
        "response_json": responseJSON.schema()
    }