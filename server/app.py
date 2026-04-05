# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the protein folding OpenEnv environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

import sys
import os

# Add the current directory to sys.path so it can find models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ProteinAction, ProteinObservation
from server.xero_environment import ProteinFoldingEnvironment


app = create_app(
    ProteinFoldingEnvironment,
    ProteinAction,
    ProteinObservation,
    env_name="xero",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the protein folding server directly."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
