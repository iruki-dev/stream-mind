import statistics

from stream_mind.bench.gate1 import gate_reasons, run_grid, run_one, summarize


def mean_score(game, mode, pressure, seeds=4):
    return statistics.mean(run_one(game, mode, s, pressure, frames=60).score for s in range(seeds))


def test_low_pressure_stream_matches_turn():
    for game in ("freeway", "snake"):
        assert mean_score(game, "stream", 0.25) == mean_score(game, "turn", 0.25)


def test_high_pressure_stream_beats_turn_and_no_fork():
    for game in ("freeway", "snake"):
        stream = mean_score(game, "stream", 4.0)
        assert stream > mean_score(game, "turn", 4.0)
        assert stream > mean_score(game, "no-fork", 4.0)


def test_event_driven_spends_fewer_tokens_than_fixed_tick():
    stream = run_one("freeway", "stream", 0, 0.25, frames=40)
    fixed = run_one("freeway", "fixed-tick", 0, 0.25, frames=40)
    assert stream.tokens * 5 < fixed.tokens
    assert stream.score == fixed.score


def test_summary_reports_gate():
    results = run_grid(seeds=2, games=("freeway",), modes=("stream", "turn"),
                       pressures={"0.25": 0.25, "4": 4.0}, frames=40)
    table, passed = summarize(results, {"0.25": 0.25, "4": 4.0})
    assert "관문 1" in table and passed


def test_gate_reasons():
    assert gate_reasons("g", [(0.25, 0.0), (2.0, 1.0)]) == []
    assert gate_reasons("g", [(0.25, -0.5), (2.0, 1.0)])
    assert gate_reasons("g", [(0.25, 0.0), (2.0, 0.0)])
