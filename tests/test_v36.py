"""3.6 Phase 0: the Gaussian head, multi-label hinge, labelled triplets and L-BFGS."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_bayes import _infer, _module, _run, n  # noqa: E402
from test_bayes import chain as design  # noqa: E402
from verify_training import chain  # noqa: E402

from ai_made_easy.core.codegen import CodegenError  # noqa: E402
from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402

TRAINER = ("train.trainer", {"epochs": 6, "batch_size": 32, "device": "cpu", "seed": 0})


def _exec(graph, tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    code = generate_training(graph, "pytorch")
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, f"<{graph.name}>", "exec"), ns)
    ns["main"]()
    return code


def _test_metric(out: str, key: str) -> float:
    return float(out.split("test:")[-1].split(f"{key}=")[1].split()[0])


# ------------------------------------------------------------------ Gaussian head

def test_gaussian_head_is_a_one_component_mixture():
    from ai_made_easy.core.bayes.codegen import probabilistic_context

    g = Graph.from_dict(design("4", [("core.dense", {"units": 8}),
                                     ("bayes.gaussian_head", {"targets": 2})],
                               [n("loss", "train.loss_mse")]))
    assert probabilistic_context(g)["mdn"] == {"k": 1, "d": 2}
    assert g.infer_shapes()["l1"] == [5]                 # weight, 2 means, 2 log-sigmas
    from ai_made_easy.core.summary import summarize

    rows = {r.type_id: r.params for r in summarize(g).layers}
    assert rows["bayes.gaussian_head"] == 8 * 5 + 5
    wrong = design("4", [("bayes.gaussian_head", {})], [n("loss", "train.loss_cross_entropy")])
    assert any("real-valued" in i.message for i in Graph.from_dict(wrong).validate())


def test_gaussian_head_likelihood_is_torchs_gaussian_nll(tmp_path):
    torch = pytest.importorskip("torch")
    data = design("4", [("core.dense", {"units": 8}), ("bayes.gaussian_head", {"targets": 2})],
                  [n("data", "data.synthetic", kind="regression", n_features=4, n_samples=100),
                   n("loss", "train.loss_mse")])
    m = _module(data, tmp_path)
    out, y = torch.randn(6, 5), torch.randn(6, 2)
    _, mu, log_sigma = m.mdn_split(out)
    var = (2 * log_sigma[:, 0]).exp()
    want = torch.nn.GaussianNLLLoss(full=True, reduction="sum")(mu[:, 0], y, var) / 6
    assert float(m.mdn_nll(out, y)) == pytest.approx(float(want), rel=1e-5)


def test_gaussian_head_learns_where_the_noise_is(tmp_path):
    torch = pytest.importorskip("torch")
    data = design("1", [("core.dense", {"units": 32}), ("core.relu", {}),
                        ("bayes.gaussian_head", {})],
                  [n("data", "data.synthetic", kind="regression", n_features=1, n_samples=100),
                   n("loss", "train.loss_mse")])
    m = _module(data, tmp_path)
    torch.manual_seed(0)
    x = torch.rand(3000, 1)
    y = 2 * x + torch.randn(3000, 1) * (0.05 + x)          # noisier on the right
    net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), m.MDNHead(32, 1, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(600):
        opt.zero_grad()
        m.mdn_nll(net(x), y).backward()
        opt.step()
    with torch.no_grad():
        mean, std = m.mdn_moments(net(torch.tensor([[0.1], [0.9]])).numpy())
    assert std[1, 0] > 2.5 * std[0, 0], std
    assert mean[0, 0] == pytest.approx(0.2, abs=0.1) and mean[1, 0] == pytest.approx(1.8, abs=0.2)


def test_gaussian_head_serves_a_prediction_with_its_spread(tmp_path):
    pytest.importorskip("torch")
    data = design("4", [("core.dense", {"units": 16}), ("core.relu", {}),
                        ("bayes.gaussian_head", {})],
                  [n("data", "data.synthetic", kind="regression", n_features=4, n_samples=400,
                     noise=0.3), n("loss", "train.loss_mse"),
                   n("tr", "train.trainer", epochs=5, batch_size=64)])
    run, script = _run(data, tmp_path)
    row = _infer(run, script, [[0.1] * 4])[0]
    assert row["std"] > 0 and isinstance(row["prediction"], float)


# ------------------------------------------------------------------ multi-label hinge

def test_multilabel_hinge_matches_torch():
    torch = pytest.importorskip("torch")
    from ai_made_easy.core.training import catalog as cat

    loss = eval(cat.render("train.loss_multilabel_margin", {}, "pytorch"),  # noqa: S307
                {"torch": torch})
    pred = torch.randn(5, 4)
    hot = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 0, 0],
                        [1, 0, 0, 1]], dtype=torch.float32)
    idx = torch.full((5, 4), -1)
    for i, row in enumerate(hot):
        on = row.nonzero().reshape(-1)
        idx[i, :len(on)] = on
    want = torch.nn.MultiLabelMarginLoss()(pred, idx)
    assert float(loss(pred, hot)) == pytest.approx(float(want), rel=1e-5)


def test_multilabel_hinge_keras_matches_torch():
    torch = pytest.importorskip("torch")
    keras = pytest.importorskip("keras")
    from ai_made_easy.core.training import catalog as cat

    t = eval(cat.render("train.loss_multilabel_margin", {}, "pytorch"), {"torch": torch})  # noqa: S307
    k = eval(cat.render("train.loss_multilabel_margin", {}, "keras"), {"ops": keras.ops})  # noqa: S307
    pred = torch.randn(4, 3)
    hot = (torch.rand(4, 3) > 0.5).float()
    got = float(keras.ops.convert_to_numpy(k(hot.numpy(), pred.numpy())))
    assert got == pytest.approx(float(t(pred, hot)), rel=1e-5)


# ------------------------------------------------------------------ labelled triplets

def _triplets_csv(path: Path, n_rows: int = 600) -> Path:
    import numpy as np

    rng = np.random.default_rng(0)
    centres = rng.normal(scale=3.0, size=(4, 4))
    ka = rng.integers(0, 4, n_rows)
    kn = (ka + rng.integers(1, 4, n_rows)) % 4
    items = [centres[k] + rng.normal(scale=0.6, size=(n_rows, 4)) for k in (ka, ka, kn)]
    cols = [f"{pre}f{i}" for pre in ("a_", "p_", "n_") for i in range(4)]
    rows = [",".join(cols)]
    rows += [",".join(str(v) for item in items for v in item[r]) for r in range(n_rows)]
    path.write_text("\n".join(rows))
    return path


@pytest.mark.parametrize("from_file", [True, False])
def test_labelled_triplets_train_an_encoder(from_file, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    width = 4 if from_file else 8
    params = {"path": str(_triplets_csv(tmp_path / "t.csv"))} if from_file else {}
    graph = chain("triplets", str(width), [("core.dense", {"units": 16}), ("core.relu", {}),
                                           ("core.dense", {"units": 8})],
                  [("data.triplets", params), ("train.loss_triplet", {"classes": 2}),
                   ("train.adam", {"lr": 0.01}), ("eval.accuracy", {}),
                   ("train.trainer", {**TRAINER[1], "epochs": 8})])
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    code = _exec(graph, tmp_path, monkeypatch)
    assert "reshape(x.shape[0], x.shape[1], -1)" in code
    out = capsys.readouterr().out
    assert _test_metric(out, "accuracy") > 0.9, out[-1500:]
    assert json.loads((tmp_path / "classes.json").read_text()) == ["negative closer",
                                                                   "positive closer"]


def test_labelled_triplets_need_the_triplet_loss():
    g = chain("triplets_ce", "8", [("core.dense", {"units": 2})],
              [("data.triplets", {}), ("train.loss_cross_entropy", {}), TRAINER])
    with pytest.raises(CodegenError, match="Triplet loss"):
        generate_training(g, "pytorch")


# ------------------------------------------------------------------ L-BFGS

def test_lbfgs_fits_a_regression(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    graph = chain("lbfgs", "6", [("core.dense", {"units": 1})],
                  [("data.synthetic", {"kind": "regression", "n_features": 6, "n_samples": 400,
                                       "noise": 0.05, "seed": 0}),
                   ("train.loss_mse", {}), ("train.lbfgs", {"max_iter": 20}),
                   ("eval.r2", {}),
                   ("train.trainer", {**TRAINER[1], "epochs": 3, "batch_size": 400})])
    code = _exec(graph, tmp_path, monkeypatch)
    assert "torch.optim.LBFGS(" in code and "optimizer.step(closure)" in code
    assert _test_metric(capsys.readouterr().out, "r2") > 0.95


def test_lbfgs_has_no_keras_form_and_family_trainers_refuse_it():
    plain = chain("lbfgs_k", "6", [("core.dense", {"units": 1})],
                  [("train.loss_mse", {}), ("train.lbfgs", {}), TRAINER])
    with pytest.raises(CodegenError, match="L-BFGS"):
        generate_training(plain, "keras")
    data = json.loads((Path(__file__).resolve().parents[1] / "ai_made_easy" / "samples"
                       / "demand_forecasting.json").read_text())
    from ai_made_easy.core.training.catalog import OPTIMIZER_IDS

    data["nodes"] = [nd for nd in data["nodes"] if nd["type"] not in OPTIMIZER_IDS]
    data["nodes"].append(n("opt", "train.lbfgs"))
    with pytest.raises(CodegenError, match="standard training loop"):
        generate_training(Graph.from_dict(data), "pytorch")
