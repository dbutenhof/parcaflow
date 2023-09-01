#!/usr/bin/env python3

import enum
import os
import platform
import signal
import socket
import subprocess
import sys
import time
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


@dataclass
class Iteration:
    number: int
    name: str
    benchmark: str


class Pbench:
    def __init__(self, config: str):
        self.config = config
        self.date = datetime.now(timezone.utc)
        self.group = "default"
        self.toolset = "medium"
        self.root = Path("/opt/pbench-agent")
        self.util_dir = self.root / "util-scripts"
        self.bin_dir = self.root / "bin"
        self.run_base = Path("/var/lib/pbench-agent")
        self.name = f"parca_{config}_{self.date:%Y.%m.%dT%H.%M.%S}"
        self.run_dir = self.run_base / self.name
        self.run_dir.mkdir(parents=True)
        self.config_file = self.run_dir / "metadata.log"
        self.log_file = self.run_base / "parca.log"
        self.logger = self.log_file.open("a")
        self.logger.reconfigure(line_buffering=True, write_through=True)
        self.config_parser = None
        self.running = self.run_dir / ".running"
        self.iteration_file = self.run_dir / "iterations.lis"
        self.current_iteration: int = 0
        self.iterations: list[Iteration] = []
        self.iteration_dir = None

        # Capture upload parameters (can we get these from ARCA signals)
        self.token = None
        self.relay = None
        self.server = None

        libs = [self.root / "lib"] + list(
            self.root.glob("lib*/python3.*/site-packages")
        )
        environ = dict()
        environ["PYTHONPATH"] = ":".join(str(p) for p in libs)
        environ["PATH"] = os.environ["PATH"] + ":".join(
            [str(self.bin_dir), str(self.util_dir)]
        )
        environ["_PBENCH_AGENT_CONFIG"] = str(self.root / "config" / "pbench-agent.cfg")
        environ["pbench_run"] = str(self.run_base)
        tmp_dir = self.run_base / "tmp"
        environ["pbench_tmp"] = str(tmp_dir)
        environ["pbench_log"] = str(self.run_base / "pbench.log")
        environ["pbench_install_dir"] = str(self.root)
        environ["pbench_lib_dir"] = str(self.root / "lib")
        environ["benchmark_run_dir"] = str(self.run_dir)
        environ["_pbench_hostname"] = socket.gethostbyname(socket.gethostname())
        environ["_pbench_full_hostname"] = platform.node()
        environ["_pbench_hostname_ip"] = " ".join(
            i.ip[0] if type(i.ip) is tuple else i.ip
            for a in ifaddr.get_adapters()
            for i in a.ips
        )
        self.environ = environ
        self.log(f"INIT {self.run_dir}")

    def log(self, str):
        date = datetime.now()
        self.logger.write(f"{date:%Y-%m-%d:%H:%M} {str}\n")

    def run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        self.log(f"RUN {' '.join(cmd)!r}")
        c = subprocess.run(
            cmd, env=self.environ, text=True, capture_output=True, check=True
        )
        self.log(
            f"{cmd[0]} -> {c.returncode}:{c.stderr if c.returncode else c.stdout!r}"
        )

    def open_config_file(self):
        if not self.config_parser:
            self.config_parser = ConfigParser(interpolation=None)
            if self.config_file.exists():
                self.config_parser.read(self.config_file)

    def write_config_file(self):
        if self.config_parser:
            with self.config_file.open("w") as f:
                self.config_parser.write(f)

    def add(self, section, key, value):
        try:
            self.open_config_file()
            if not self.config_parser.has_section(section):
                self.config_parser.add_section(section)
            self.config_parser.set(section, key, value)
        except Exception as e:
            raise Exception(f"Failed setting {section}:{key} = {value!r}: {e}")

    def add_iteration(
        self, number: typing.Optional[int] = None, name: typing.Optional[str] = None
    ):
        num = number if number else self.current_iteration + 1
        nam = name if name else f"default-{num}"
        self.iterations.append(Iteration(num, nam, "Arca workflow"))
        self.add(f"iterations/{nam}", "iteration_number", str(num))
        self.add(f"iterations/{nam}", "iteration_name", nam)
        self.current_iteration = num
        self.log(f"ITERATION {num}:{nam}")
        self.iteration_dir = self.run_dir / nam / "sample1"
        self.write_config_file()

    def done(self):
        self.log("DONE")
        self.config_parser = None
        self.logger.close()


@dataclass
class MainInputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """

    """Selected performance tool set (light, medium, heavy)"""
    toolset: typing.Annotated[
        typing.Optional[str],
        schema.name("toolset"),
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

    server: typing.Annotated[
        typing.Optional[str],
        schema.name("server"),
        schema.example("https://localhost:8443"),
        #        schema.required_if_not("relay"),
        #        schema.conflicts("relay"),
    ] = None

    relay: typing.Annotated[
        typing.Optional[str],
        schema.name("relay"),
        schema.example("http://localhost:8080/RELAY_ID"),
        #        schema.required_if_not("server"),
        #        schema.conflicts("server"),
    ] = None

    token: typing.Annotated[typing.Optional[str], schema.name("token")] = None


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


@dataclass
class SuccessOutput:
    """
    This is the output data structure for the success case.
    """

    status: int
    message: str


@dataclass
class ErrorOutput:
    """
    This is the output data structure in the error  case.
    """

    error: str


class Terminated(Exception):
    pass


class Finalize(Exception):
    pass


class Upload(Exception):
    pass


class Advanced(Exception):
    pass


class Done(Exception):
    pass


class State(enum.Enum):
    BEGIN = enum.auto()
    INITIALIZE = enum.auto()
    START = enum.auto()
    STOP = enum.auto()
    FINALIZE = enum.auto()
    UPLOAD = enum.auto()
    TERMINATE = enum.auto()
    ERROR = enum.auto()


def wait_for_signal(state: State, pbench: Pbench) -> typing.Optional[State]:
    pbench.log("Waiting for something to happen")
    try:
        signal.signal(signal.SIGHUP, advance_handler)
        signal.signal(signal.SIGUSR1, finalize_handler)
        signal.signal(signal.SIGUSR2, upload_handler)
        signal.signal(signal.SIGTERM, terminate_handler)
        while True:
            pbench.log("sleep... (zzzz)")
            time.sleep(10.0 * 60.0)
            pbench.log("woke... (I'm tired)")
    except Terminated:
        pbench.log("SIGTERM")
        next = State.TERMINATE
    except Advanced:
        pbench.log("SIGHUP")
        next = transitions[state].next
    except Finalize:
        pbench.log("SIGUSR1")
        next = State.FINALIZE
    except Upload:
        pbench.log("SIGUSR2")
        next = State.UPLOAD

    signal.signal(signal.SIGTERM, signal.SIG_BLOCK)
    signal.signal(signal.SIGUSR2, signal.SIG_BLOCK)
    signal.signal(signal.SIGUSR1, signal.SIG_BLOCK)
    signal.signal(signal.SIGHUP, signal.SIG_BLOCK)
    pbench.log(f"Something happened: returning {next}")
    return next


def initialize_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log("INITIALIZE")
    pbench.run(["pbench-register-tool-set", pbench.toolset])
    pbench.running.mkdir(parents=True)
    c = pbench.run(["pbench-tool-meister-start", "default"])
    pbench.open_config_file()
    pbench.add("pbench", "script", "parcaflow")
    pbench.add("pbench", "config", pbench.config)
    pbench.add("pbench", "date", f"{pbench.date:%Y-%m-%dT%H:%M:%S}")
    pbench.write_config_file()
    return c


def start_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log("START")
    # TODO: signal could provide values?
    pbench.add_iteration()
    return pbench.run(
        [
            "pbench-start-tools",
            "--group",
            pbench.group,
            "--dir",
            str(pbench.iteration_dir),
        ]
    )


def stop_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log("STOP")
    run_dir = str(pbench.iteration_dir)
    group = pbench.group

    pbench.run(["pbench-stop-tools", "--group", group, "--dir", run_dir])
    pbench.run(["pbench-send-tools", "--group", group, "--dir", run_dir])
    return pbench.run(["pbench-postprocess-tools", "--group", group, "--dir", run_dir])


def finalize_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log("FINALIZE")
    # "--sysinfo", "default", ... doesn't work in centos-8
    pbench.running.rmdir()
    pbench.add("pbench", "iterations", ",".join(i.name for i in pbench.iterations))
    pbench.write_config_file()
    try:
        c = pbench.run(["pbench-tool-meister-stop", "default"])
        return c
    except Exception as e:
        pbench.log(f"Tool meister stop failed with '{e}': continuing")
    return None


def upload_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log(
        f"UPLOAD relay {pbench.relay!r}, server {pbench.server!r}:{pbench.token!r}"
    )
    command = ["pbench-results-move"]
    pbench.log(f"RELAY {pbench.relay}(type {type(pbench.relay).__name__!r})")
    if not pbench.server:
        command += ["--relay", pbench.relay]
    else:
        command += ["--server", pbench.server, "--token", pbench.token]
    return pbench.run(command)


def terminate_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log("TERMINATE")
    raise Done()


def error_action(pbench: Pbench) -> subprocess.CompletedProcess:
    pbench.log("ERROR STATE")
    return pbench.run(["ls", str(pbench.run_dir)])


@dataclass
class Transition:
    action: typing.Optional[typing.Callable[[Pbench], subprocess.CompletedProcess]]
    next: typing.Optional[State]
    transitions: list[State]


transitions: dict[State, Transition] = {
    State.BEGIN: Transition(
        None, State.INITIALIZE, [State.INITIALIZE, State.TERMINATE]
    ),
    State.INITIALIZE: Transition(
        initialize_action,
        State.START,
        [State.START, State.FINALIZE, State.TERMINATE, State.ERROR],
    ),
    State.START: Transition(
        start_action, State.STOP, [State.STOP, State.TERMINATE, State.ERROR]
    ),
    State.STOP: Transition(
        stop_action,
        State.START,
        [State.START, State.FINALIZE, State.TERMINATE, State.ERROR],
    ),
    State.FINALIZE: Transition(
        finalize_action, State.TERMINATE, [State.UPLOAD, State.TERMINATE, State.ERROR]
    ),
    State.UPLOAD: Transition(
        upload_action,
        State.TERMINATE,
        [State.INITIALIZE, State.UPLOAD, State.TERMINATE],
    ),
    State.TERMINATE: Transition(terminate_action, None, [State.ERROR]),
    State.ERROR: Transition(error_action, State.ERROR, [State.TERMINATE]),
}


def finalize_handler(*args):
    raise Finalize()


def terminate_handler(*args):
    raise Terminated()


def upload_handler(*args):
    raise Upload()


def advance_handler(*args):
    raise Advanced()


@plugin.step(
    id="main",
    name="Pbench state machine",
    description="Enable collecting performance metrics across workload steps",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def main(
    params: MainInputParams,
) -> typing.Tuple[str, typing.Union[SuccessOutput, ErrorOutput]]:
    """The function is the implementation for the step. It needs the decorator
    above to make it into a step. The type hints for the params are required.

    :param params:

    :return: the string identifying which output it is, as well the output
        structure
    """

    try:
        pbench = Pbench(params.config)
    except Exception as e:
        return "error", ErrorOutput(f"Config error: '{e}'")

    pbench.relay = params.relay
    pbench.server = params.server
    pbench.token = params.token

    pbench.log("Starting MAIN")
    state = State.BEGIN
    next_state = State.INITIALIZE
    while True:
        pbench.log(f"TOP {state} -> {next_state}")
        try:
            if next_state != state:
                state = next_state
                if transitions[state].action:
                    result = transitions[state].action(pbench)
                    if result is not None:
                        pbench.log(
                            f"ACTION[{state}] -> {result.returncode}:{result.stdout!r}"
                        )
        except Done:
            pbench.log("Declaring DONE and exiting loop")
            break
        except subprocess.CalledProcessError as e:
            pbench.log(
                f"[ERROR] state {state} command failed {e.returncode}:{e.stderr!r}"
            )
        except Exception as e:
            pbench.log(f"[ERROR] state {state} unexpected error '{e}'")

        pbench.log(f"{state} is blocking for input")
        try:
            s = wait_for_signal(state, pbench)
        except Exception as e:
            pbench.log(f"WAIT raised '{e}' and I don't know why")
            continue
        if s in transitions[state].transitions:
            pbench.log(f"Transition from {state} to {s}")
            next_state = s
        else:
            pbench.log(f"State {state} cannot transition to {s}")
            continue

    pbench.log("Completing MAIN")
    pbench.done()
    return "success", SuccessOutput(result.returncode, f"DONE: stdout {result.stdout}")


if __name__ == "__main__":
    sys.exit(
        plugin.run(
            plugin.build_schema(
                # List your step functions here:
                main
            )
        )
    )
