#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import platform
from functools import wraps

import cv2
import numpy as np
import pytest
import torch

from lerobot import available_robots
from lerobot.common.utils.import_utils import is_package_available

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pass this as the first argument to init_hydra_config.
DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"

ROBOT_CONFIG_PATH_TEMPLATE = "lerobot/configs/robot/{robot}.yaml"

TEST_ROBOT_TYPES = available_robots + [f"mocked_{robot_type}" for robot_type in available_robots]


def require_x86_64_kernel(func):
    """
    Decorator that skips the test if plateform device is not an x86_64 cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if platform.machine() != "x86_64":
            pytest.skip("requires x86_64 plateform")
        return func(*args, **kwargs)

    return wrapper


def require_cpu(func):
    """
    Decorator that skips the test if device is not cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEVICE != "cpu":
            pytest.skip("requires cpu")
        return func(*args, **kwargs)

    return wrapper


def require_cuda(func):
    """
    Decorator that skips the test if cuda is not available.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("requires cuda")
        return func(*args, **kwargs)

    return wrapper


def require_env(func):
    """
    Decorator that skips the test if the required environment package is not installed.
    As it need 'env_name' in args, it also checks whether it is provided as an argument.
    If 'env_name' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'env_name' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "env_name" in arg_names:
            # Get the index of 'env_name' and retrieve the value from args
            index = arg_names.index("env_name")
            env_name = args[index] if len(args) > index else kwargs.get("env_name")
        else:
            raise ValueError("Function does not have 'env_name' as an argument.")

        # Perform the package check
        package_name = f"gym_{env_name}"
        if env_name is not None and not is_package_available(package_name):
            pytest.skip(f"gym-{env_name} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package_arg(func):
    """
    Decorator that skips the test if the required package is not installed.
    This is similar to `require_env` but more general in that it can check any package (not just environments).
    As it need 'required_packages' in args, it also checks whether it is provided as an argument.
    If 'required_packages' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'required_packages' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "required_packages" in arg_names:
            # Get the index of 'required_packages' and retrieve the value from args
            index = arg_names.index("required_packages")
            required_packages = args[index] if len(args) > index else kwargs.get("required_packages")
        else:
            raise ValueError("Function does not have 'required_packages' as an argument.")

        if required_packages is None:
            return func(*args, **kwargs)

        # Perform the package check
        for package in required_packages:
            if not is_package_available(package):
                pytest.skip(f"{package} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package(package_name):
    """
    Decorator that skips the test if the specified package is not installed.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_package_available(package_name):
                pytest.skip(f"{package_name} not installed")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_robot(func):
    """
    Decorator that skips the test if a robot is not available

    The decorated function must have two arguments `request` and `robot_type`.

    Example of usage:
    ```python
    @pytest.mark.parametrize(
        "robot_type", ["koch", "aloha"]
    )
    @require_robot
    def test_require_robot(request, robot_type):
        pass
    ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Access the pytest request context to get the is_robot_available fixture
        request = kwargs.get("request")
        robot_type = kwargs.get("robot_type")

        if request is None:
            raise ValueError("The 'request' fixture must be an argument of the test function.")

        # Run test with a monkeypatched version of the robot devices.
        if robot_type.startswith("mocked_"):
            kwargs["robot_type"] = robot_type.replace("mocked_", "")

            monkeypatch = request.getfixturevalue("monkeypatch")
            monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)

        # Run test with a real robot. Skip test if robot connection fails.
        else:
            # `is_robot_available` is defined in `tests/conftest.py`
            if not request.getfixturevalue("is_robot_available"):
                pytest.skip(f"A {robot_type} robot is not available.")

        return func(*args, **kwargs)

    return wrapper


class MockVideoCapture(cv2.VideoCapture):
    image = {
        "480x640": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "720x1280": np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8),
    }

    def __init__(self, *args, **kwargs):
        self._mock_dict = {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }

    def isOpened(self):  # noqa: N802
        return True

    def set(self, propId: int, value: float) -> bool:  # noqa: N803
        self._mock_dict[propId] = value
        return True

    def get(self, propId: int) -> float:  # noqa: N803
        value = self._mock_dict[propId]
        if value == 0:
            if propId == cv2.CAP_PROP_FRAME_HEIGHT:
                value = 480
            elif propId == cv2.CAP_PROP_FRAME_WIDTH:
                value = 640
        return value

    def read(self):
        h = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = self.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret = True
        return ret, self.image[f"{h}x{w}"]

    def release(self):
        pass
