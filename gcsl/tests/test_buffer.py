import gcsl
from itertools import chain


def test_buffer():
    b = gcsl.ExperienceBuffer(10)
    t = [(0, 1, 5, 2), (1, 1, 5, 1), (2, 1, 5, 0)]
    b.insert([t, t, t])
    assert list(b.memory) == list(chain(*[t, t, t]))
    b.insert([t])
    assert list(b.memory) == list(
        chain(
            *[
                [(2, 1, 5, 0)],
                t,
                t,
                t,
            ]
        )
    )


def test_sample():
    b = gcsl.ExperienceBuffer(100)
    # s,a,g,h
    t1 = [(1, 1, 5, 4), (2, 1, 5, 3), (3, 1, 5, 2), (4, 1, 5, 1), (5, 1, 5, 0)]
    t2 = [
        (-1, -1, -5, 4),
        (-2, -1, -5, 3),
        (-3, -1, -5, 2),
        (-4, -1, -5, 1),
        (-5, -1, -5, 0),
    ]
    b.insert([t1, t2])

    def relabel(x, y):
        return y[0]

    samples = b.sample(100, relabel, max_horizon=None)
    for s, a, g, h in samples:
        assert g == s + h * a

    samples = b.sample(100, relabel, max_horizon=2)
    for s, a, g, h in samples:
        assert h <= 2
        assert g == s + h * a

    samples = gcsl.sample_buffers([b, b], 100, relabel, max_horizon=2)
    for s, a, g, h in samples:
        assert h <= 2
        assert g == s + h * a