import random

import pytest
import torch

from bioner.mtl import assign_flat_grad, flat_grad, make_schedule, pcgrad, sampling_probs

SIZES = {"BC4CHEMD": 31000, "NCBI": 5000, "BC5CDR": 4500}


def test_proportional_matches_sizes():
    p = sampling_probs(SIZES, "proportional")
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["BC4CHEMD"] == pytest.approx(31000 / sum(SIZES.values()))


def test_uniform_is_flat():
    p = sampling_probs(SIZES, "uniform")
    assert all(v == pytest.approx(1 / 3) for v in p.values())


def test_temperature_interpolates_between_proportional_and_uniform():
    prop = sampling_probs(SIZES, "proportional")["BC4CHEMD"]
    t1 = sampling_probs(SIZES, "temperature", 1.0)["BC4CHEMD"]
    t2 = sampling_probs(SIZES, "temperature", 2.0)["BC4CHEMD"]
    t100 = sampling_probs(SIZES, "temperature", 100.0)["BC4CHEMD"]
    assert t1 == pytest.approx(prop)
    assert 1 / 3 < t100 < t2 < t1
    assert t100 == pytest.approx(1 / 3, abs=0.01)


def test_temperature_shrinks_the_dominant_corpus_but_keeps_order():
    p = sampling_probs(SIZES, "temperature", 5.0)
    assert p["BC4CHEMD"] > p["NCBI"] > p["BC5CDR"]


@pytest.mark.parametrize("bad", [{}, {"a": 0}, {"a": -1}])
def test_bad_sizes_rejected(bad):
    with pytest.raises(ValueError):
        sampling_probs(bad)


def test_unknown_sampling_and_bad_temperature():
    with pytest.raises(ValueError):
        sampling_probs(SIZES, "nope")
    with pytest.raises(ValueError):
        sampling_probs(SIZES, "temperature", 0)


def test_schedule_follows_probs_and_is_seeded():
    probs = {"a": 0.8, "b": 0.2}
    s1 = make_schedule(probs, 5000, random.Random(1))
    s2 = make_schedule(probs, 5000, random.Random(1))
    assert s1 == s2
    assert s1.count("a") / 5000 == pytest.approx(0.8, abs=0.03)


def test_pcgrad_projects_conflicting_gradients():
    g1, g2 = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])  # dot = -1 < 0
    out = pcgrad([g1, g2], random.Random(0))
    # g1' = g1 + 0.5*g2 = [.5,.5]; g2' = g2 + g1 = [0,1]; mean = [.25,.75]
    assert torch.allclose(out, torch.tensor([0.25, 0.75]))


def test_pcgrad_equals_plain_mean_when_no_conflict():
    gs = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]
    assert torch.allclose(pcgrad(gs, random.Random(0)), torch.stack(gs).mean(0))


def test_pcgrad_single_task_is_identity():
    g = torch.tensor([1.0, -2.0])
    assert torch.equal(pcgrad([g], random.Random(0)), g)


def test_pcgrad_does_not_mutate_inputs_and_handles_zero_gradient():
    g1, g2 = torch.tensor([1.0, 0.0]), torch.zeros(2)
    a, b = g1.clone(), g2.clone()
    out = pcgrad([g1, g2], random.Random(0))
    assert torch.equal(g1, a) and torch.equal(g2, b)
    assert torch.isfinite(out).all()


def test_flat_grad_round_trip():
    lin = torch.nn.Linear(3, 2)
    lin.weight.grad = torch.arange(6.0).view(2, 3)
    lin.bias.grad = None  # missing grads count as zeros
    flat = flat_grad(list(lin.parameters()))
    assert flat.tolist() == [0, 1, 2, 3, 4, 5, 0, 0]
    assign_flat_grad(list(lin.parameters()), flat * 2)
    assert lin.weight.grad.tolist() == [[0, 2, 4], [6, 8, 10]]
    assert lin.bias.grad.tolist() == [0, 0]
