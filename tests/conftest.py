import pytest

from ashdpf.config import ASHDPFConfig
from ashdpf.experiments import make_world
from ashdpf.models import EventVector, FailureEvent


def make_event(job="p1.t1", pipeline="p1", logs=("Task exited with return code 1",),
               cat=None, ts=100.0, eid="e1", **meta):
    return FailureEvent(eid, pipeline, job, ts,
                        EventVector(list(logs), dict(meta), {}, [], {}), cat)


@pytest.fixture
def event_factory():
    return make_event


@pytest.fixture(scope="session")
def small_world():
    """A small deterministic synthetic world shared by integration tests."""
    return make_world(seed=11, n_history=200, n_events=500, n_pipelines=60, n_groups=6)


@pytest.fixture
def cfg():
    return ASHDPFConfig(min_train_samples=10, retrain_every=50)
