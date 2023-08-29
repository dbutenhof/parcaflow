#!/usr/bin/env python3

from enum import Enum
import enum
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import typing
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ifaddr
from arcaflow_plugin_sdk import plugin, schema, validation

"""
Prototype Pbench Agent plugin for Arcaflow

What inputs?
* tool set
* individual tools
* remote nodes? How do we sync this with Arca workflow deployment? (Can we read
  that?) I suppose the first pass is just to "do something" locally
"""


class Podman:
    def __init__(
        self, name: str, image: str, environ: typing.Optional[dict[str, str]] = None
    ):
        self.name = name
        self.image = image
        self.environ = [f"-e={n}={v}" for n, v in environ.items()]

    def _remote(self, cmd: list[str]) -> subprocess.CompletedProcess:
        podcmd = ["podman-remote"] + cmd
        return subprocess.run(podcmd, text=True, capture_output=True, check=True)

    def run(self) -> subprocess.CompletedProcess:
        return self._remote(
            [
                "run",
                "--entrypoint=/sbin/init",
                "--network=host",
                "--detach",
                "--rm",
                "--name",
                self.name,
                self.image,
            ]
        )

    def exec(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return self._remote(
            ["exec", "--interactive"]
            + self.environ
            + [self.name, "/container_entrypoint"]
            + cmd
        )

    def kill(self) -> subprocess.CompletedProcess:
        return self._remote(["stop", self.name])


@dataclass
class Iteration:
    number: int
    name: str
    benchmark: str


class Config:
    def __init__(self, config: str):
        self.config = None
        self.root = Path("/opt/pbench-agent")
        self.util_dir = self.root / "util-scripts"
        self.run_base = Path("/var/lib/pbench_agent")
        self.name = f"parca_{config}_{datetime.now(timezone.utc):%Y.%m.%dT%H.%M.%S}"
        self.run_dir = self.run_base / self.name
        self.config_file = self.run_dir / "metadata.log"
        self.running = self.run_dir / ".running"
        self.iteration_file = self.run_dir / "iterations.lis"
        self.current_iteration: typing.Optional[int] = None
        self.iterations: list[Iteration] = []
        self.environ: dict[str, str] = {}

    def add(self, section, key, value):
        if not self.config:
            self.config = ConfigParser(interpolation=None)
            if self.config_file.exists():
                self.config.read(self.config_file)
        self.config.set(section, key, value)

    def add_iteration(self, number: int, name: str):
        self.iterations.append(Iteration(number, name, "Arca workflow"))
        self.add(f"iterations/{name}", "iteration_number", str(number))
        self.add(f"iterations/{name}", "iteration_name", name)
        self.add(f"iterations/{name}", "iteration_number", "arca workflow")
        self.current_iteration = number

    def done(self):
        if self.config:
            with self.config_file.open("w") as f:
                self.config.write(f)
        self.config = None


@dataclass
class InitInputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """

    """Selected performance tool set (light, medium, heavy)"""
    toolset: typing.Annotated[
        typing.Optional[str],
        schema.name("index"),
        validation.min(1),
        schema.description("The toolset name to register (default 'medium')"),
    ] = "medium"

    """Configuration string (included in dataset name and in metadata)"""
    config: typing.Annotated[
        typing.Optional[str],
        schema.name("config"),
        validation.min(1),
        schema.description("Informational string for dataset"),
    ] = "parca"


@dataclass
class BeginInputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """

    name: typing.Annotated[
        str, schema.name("name"), schema.description("Iteration name")
    ]
    number: typing.Annotated[
        int, schema.name("number"), schema.description("Iteration number")
    ]
    hook: typing.Annotated[typing.Optional[str], schema.name("hook")] = None


@dataclass
class EndInputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """

    hook: typing.Annotated[typing.Optional[str], schema.name("hook")] = None


@dataclass
class FinalInputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """

    hook: typing.Annotated[typing.Optional[str], schema.name("hook")] = None


@dataclass
class UploadInputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """

    hook: typing.Annotated[typing.Optional[str], schema.name("hook")] = None
    server: typing.Annotated[
        typing.Optional[str],
        schema.name("server"),
        schema.min(1),
        schema.example("https://localhost:8443"),
        #        schema.required_if_not("relay"),
        #        schema.conflicts("relay"),
    ] = None
    relay: typing.Annotated[
        typing.Optional[str],
        schema.name("relay"),
        schema.min(1),
        schema.example("http://localhost:8080/RELAY_ID"),
        #        schema.required_if_not("server"),
        #        schema.conflicts("server"),
    ] = None
    token: typing.Annotated[
        typing.Optional[str], schema.name("token"), schema.min(1)
    ] = None


@dataclass
class SuccessOutput:
    """
    This is the output data structure for the success case.
    """

    message: str


@dataclass
class ErrorOutput:
    """
    This is the output data structure in the error  case.
    """

    error: str


@dataclass
class InitializeOutput:
    """Initialize outputs a bunch of context consumed by the other steps"""

    run_name: str
    environment: dict[str, str]
    message: str


class State(enum.Enum):
    INITIALIZE = enum.auto()
    START = enum.auto()
    STOP = enum.auto()
    FINALIZE = enum.auto()
    UPLOAD = enum.auto()
    TERMINATE = enum.auto()


@plugin.step(
    id="pbench",
    name="Pbench state machine",
    description="Enable collecting performance metrics across workload steps",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def pbench(
    params: InitInputParams,
) -> typing.Tuple[str, typing.Union[InitializeOutput, ErrorOutput]]:
    """The function is the implementation for the step. It needs the decorator
    above to make it into a step. The type hints for the params are required.

    :param params:

    :return: the string identifying which output it is, as well the output
        structure
    """

    config = Config(params.config)
    container_name = "parcaflowbench"
    environ = dict()
    environ["pbench_run"] = str(config.run_base)
    tmp_dir = config.run_base / "tmp"
    environ["pbench_tmp"] = str(tmp_dir)
    environ["pbench_log"] = str(config.run_base / "pbench.log")
    environ["pbench_install_dir"] = str(config.root)
    environ["pbench_lib_dir"] = str(config.root / "lib")
    environ["benchmark_run_dir"] = str(config.run_dir)
    environ["_pbench_hostname"] = socket.gethostbyname(socket.gethostname())
    environ["_pbench_full_hostname"] = platform.node()
    environ["_pbench_hostname_ip"] = " ".join(
        i.ip[0] if type(i.ip) is tuple else i.ip
        for a in ifaddr.get_adapters()
        for i in a.ips
    )

    bash = shutil.which("bash")
    with subprocess.Popen(
        [str(bash)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
    ) as gofer:
        state = State.INITIALIZE
        while state != State.TERMINATE:
            try:
                gofer.stdin.write(f"mkdir -p {config.run_dir}")
                # podman = Podman(
                #     container_name,
                #     "quay.io/pbench/pbench-agent-all-fedora-38:main",
                #     environ,
                # )
                # podman.run()
                # podman.exec(["mkdir", "-p", str(config.run_dir)])
                # podman.exec(["pbench-register-tool-set"])
                # podman.exec(["mkdir", "-p", str(config.running)])
                # x = podman.exec(["ls", "-lRh", str(config.run_base)]).stdout
                # print(x)
                # start = podman.exec(["pbench-tool-meister-start", "default"])
            except subprocess.CalledProcessError as e:
                return "error", ErrorOutput(
                    f"{e.cmd} failed with {e.returncode}: {e.stderr!r}"
                )

    return "success", InitializeOutput(
        config.name, environ, f"DONE: stdout {start.stdout}"
    )


@plugin.step(
    id="begin",
    name="Begin measuring",
    description="Begin a measuring interval",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def begin(
    params: BeginInputParams,
) -> typing.Tuple[str, typing.Union[SuccessOutput, ErrorOutput]]:
    """The function is the implementation for the step. It needs the decorator
    above to make it into a step. The type hints for the params are required.

    :param params:

    :return: the string identifying which output it is, as well the output
        structure
    """

    logger = logging.getLogger("begin")
    run_context = RunContext.get_context()
    if not run_context:
        logger.error("Didn't find run context")
        return "error", ErrorOutput(f"BEGIN run_context is missing")
    run_context.add_iteration(params.number, params.name)

    try:
        start_tools = podman_remote(
            ["pbench-start-tools", "--group", "default", "--dir", run_context.run_dir]
        )
    except subprocess.CalledProcessError as e:
        return "error", ErrorOutput(f"{e.cmd} failed with {e.returncode}: {e.stderr!r}")

    return "success", SuccessOutput(start_tools.stdout)


@plugin.step(
    id="end",
    name="Terminate measuring",
    description="End a measuring interval",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def end(
    params: EndInputParams,
) -> typing.Tuple[str, typing.Union[SuccessOutput, ErrorOutput]]:
    """The function is the implementation for the step. It needs the decorator
    above to make it into a step. The type hints for the params are required.

    :param params:

    :return: the string identifying which output it is, as well the output
        structure
    """

    logger = logging.getLogger("end")
    run_context = RunContext.get_context()
    if not run_context:
        logger.error("Didn't find run context")
        return "error", ErrorOutput(f"END run_context is missing")
    run_dir = str(run_context.run_dir)
    group = run_context.group

    try:
        podman_remote(["pbench-stop-tools", "--group", group, "--dir", run_dir])
        podman_remote(["pbench-send-tools", "--group", group, "--dir", run_dir])
        post = podman_remote(
            ["pbench-postprocess-tools", "--group", group, "--dir", run_dir]
        )
    except subprocess.CalledProcessError as e:
        return "error", ErrorOutput(f"{e.cmd} failed with {e.returncode}: {e.stderr!r}")

    return "success", SuccessOutput(post.stdout)


@plugin.step(
    id="finalize",
    name="Finalize Tool Meister",
    description="Shut down Tool Meister and collect results",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def finalize(
    params: FinalInputParams,
) -> typing.Tuple[str, typing.Union[SuccessOutput, ErrorOutput]]:
    """The function is the implementation for the step. It needs the decorator
    above to make it into a step. The type hints for the params are required.

    :param params:

    :return: the string identifying which output it is, as well the output
        structure
    """

    logger = logging.getLogger("finalize")

    try:
        stop = podman_remote(
            ["pbench-tool-meister-stop" "--sysinfo", "default", "default"]
        )
    except subprocess.CalledProcessError as e:
        return "error", ErrorOutput(f"{e.cmd} failed with {e.returncode}: {e.stderr!r}")

    run_context.running.rmdir()

    return "success", SuccessOutput(stop.stdout)


@plugin.step(
    id="upload",
    name="Upload results",
    description="Upload results to Pbench Server or Relay",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def upload(
    params: UploadInputParams,
) -> typing.Tuple[str, typing.Union[SuccessOutput, ErrorOutput]]:
    """The function is the implementation for the step. It needs the decorator
    above to make it into a step. The type hints for the params are required.

    :param params:

    :return: the string identifying which output it is, as well the output
        structure
    """

    logger = logging.getLogger("initialize")
    run_context = RunContext.get_context()
    if not run_context:
        logger.error("Didn't find run context")
        return "error", ErrorOutput(f"UPLOAD run_context is missing")

    try:
        command = ["pbench-results-move"]
        if params.relay:
            command += ["--relay", params.relay]
        else:
            command += ["--server", params.server, "--token", params.token]
        upload = podman_remote(command)
    except subprocess.CalledProcessError as e:
        return "error", ErrorOutput(f"{e.cmd} failed with {e.returncode}: {e.stderr!r}")

    return "success", SuccessOutput(upload.stdout)


if __name__ == "__main__":
    sys.exit(
        plugin.run(
            plugin.build_schema(
                # List your step functions here:
                initialize,
                begin,
                end,
                finalize,
                upload,
            )
        )
    )
