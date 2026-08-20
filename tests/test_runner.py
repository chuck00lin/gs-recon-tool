"""Stopping a run has to actually stop it -- containers included."""

import pathlib
import threading
import time

from gs_recon.config import Config
from gs_recon.pipeline import Plan, ProjectPlan
from gs_recon.runner import Runner, name_docker_run
from gs_recon.stages import Step, build_sfm_steps, docker_base

PROJECT = pathlib.Path("/data/plantA-frames")


def _plan(*argvs: list[str]) -> Plan:
    steps = [
        Step(key=f"t{i}", label=f"step {i}", stage="splat", project=PROJECT, argv=argv)
        for i, argv in enumerate(argvs)
    ]
    return Plan(projects=[ProjectPlan(project=PROJECT, video=None, steps=steps)])


# -- container plumbing ------------------------------------------------------
def test_containers_get_an_init_process():
    """Without --init, PID 1 ignores SIGTERM and the container never stops."""
    argv = docker_base("some/image", [(PROJECT, "/ws", "rw")])
    assert argv[:4] == ["docker", "run", "--rm", "--init"]


def test_every_docker_step_can_be_killed_by_name():
    docker_steps = 0
    for step in build_sfm_steps(Config(), PROJECT):
        argv, container = name_docker_run(step.exec_argv(), "gsr-test-1")
        if step.argv[0] != "docker":
            assert container is None
            continue
        docker_steps += 1
        assert container == "gsr-test-1"
        assert argv[:4] == ["docker", "run", "--name", "gsr-test-1"]
    assert docker_steps


def test_naming_leaves_non_docker_commands_alone():
    argv = ["python", "-m", "gs_recon.tools.frame_extract", "extract"]
    assert name_docker_run(argv, "gsr-test-1") == (argv, None)


def test_naming_leaves_docker_subcommands_alone():
    argv = ["docker", "ps", "-a"]
    assert name_docker_run(argv, "gsr-test-1") == (argv, None)


# -- stopping ----------------------------------------------------------------
def test_stop_kills_the_command_in_flight():
    runner = Runner(_plan(["sleep", "60"], ["sleep", "60"]), on_log=lambda line: None)
    result: list = []
    worker = threading.Thread(target=lambda: result.append(runner.run()))
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while runner._proc is None and time.monotonic() < deadline:
            time.sleep(0.02)
        runner.stop()
        worker.join(timeout=15)
    finally:
        runner.stop()

    assert not worker.is_alive(), "run() did not return after stop()"
    assert result and result[0].stopped is True
    assert result[0].failed_index == 0          # the second step never started


def test_stop_before_the_run_starts_is_honoured():
    runner = Runner(_plan(["sleep", "60"]), on_log=lambda line: None)
    runner.stop()
    result = runner.run()
    assert result.stopped is True
    assert runner.stopping is True
