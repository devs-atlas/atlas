import os
from typing import Dict, List, Union

from fastapi import FastAPI
from llama_index.graph_stores import NebulaGraphStore

app = FastAPI()

network_dict: Dict[str, List[List[str]]] = {"user": [["1"]]}

@app.get("/api")
def read_root():
    return network_dict