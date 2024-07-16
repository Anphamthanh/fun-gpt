import json
from typing import Any
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return super(NumpyEncoder, self).default(obj)
        except Exception as e:
            print(f"Error in NumpyEncoder: {str(e)}")
            print(f"Object: {obj}")
            raise e
