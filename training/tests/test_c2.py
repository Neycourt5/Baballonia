"""Mechanics fixtures only. These tests do not establish personal tracking quality."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from babble_personal import c2
from babble_personal.schema import EXPRESSION_NAMES


def context(nonidentity=False):
    return dict(Ranges=[[.2, .8, 0, 1] if nonidentity else [0, 1, 0, 1] for _ in range(45)],
                EffectiveJawExponent=1, FilterEnabled=True, MinCutoff=.5, Beta=3)


def row(value=.25, phase="hold", dims=(4,)):
    target = np.zeros(45).tolist()
    for d in dims: target[d] = value
    return {"cue": {"phase": phase, "dims": list(dims), "target": target}}


def test_unknown_is_not_zero_and_closed_lips_does_not_imply_jaw():
    target, weight, evidence = c2.labels([row(0, dims=(18,))], context(), True)
    assert weight[0, 18] > 0 and evidence[0, 18] == "explicit_absent"
    assert weight[0, 4] == 0 and evidence[0, 4] == "unknown"
    assert np.count_nonzero(weight) == 1


@pytest.mark.parametrize("phase", ["prepare", "settling", "rest", "transition", "ramp", "ramp_up", "ramp_down"])
def test_transitions_and_legacy_rest_never_supervise(phase):
    assert not c2.labels([row(phase=phase)], context(), True)[1].any()


def test_legacy_and_unaccepted_are_preservation_only():
    assert not c2.labels([row()], context(), True, legacy=True)[1].any()
    assert not c2.labels([row()], context(), False)[1].any()


def test_nonidentity_targets_are_inverted_once():
    target, weight, _ = c2.labels([row(.25, dims=(4, 19, 20))], context(True), True)
    assert target[0, 4] == pytest.approx(.35)
    assert c2.emit(target, context(True))[0, 4] == pytest.approx(.25)
    assert weight[0, 21] == 0


def test_entirely_masked_batch_has_finite_zero_human_loss():
    reference = torch.full((2, 45), .4)
    predicted = reference.clone().requires_grad_()
    target = torch.full_like(reference, float("nan"))
    loss, human, prior = c2.masked_loss(predicted, reference, target, torch.zeros_like(reference), torch.ones(45))
    loss.backward()
    assert human.item() == 0 and prior.item() == 0 and torch.isfinite(predicted.grad).all()


def test_unknown_channels_exactly_keep_personal_reference_after_learning():
    torch.manual_seed(4)
    support = np.zeros(45, np.float32); support[4] = 1
    model = c2.C2Head(support, embedding_dim=8)
    reference = torch.rand(6, 45)
    features = torch.rand(6, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(3):
        loss = (model(reference, features)[:, 4] - .8).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert torch.equal(model(reference, features)[:, 5:], reference[:, 5:])
    assert torch.equal(model(reference, features)[:, :4], reference[:, :4])


def test_split_keeps_wearing_sessions_and_check_separate():
    records = [dict(origin=o, role=r) for o,r in [("a","practice"),("a","practice"),("b","practice"),("c","check"),("legacy-unknown","replay")]]
    train, validation, check = c2.split(records)
    assert {r["origin"] for r in train} == {"a", "legacy-unknown"}
    assert {r["origin"] for r in validation} == {"b"}
    assert {r["origin"] for r in check} == {"c"}
    assert not c2.split([dict(origin="a", role="practice")])[1]


def test_positive_and_absent_coverage_both_required():
    target, weight, _ = c2.labels([row(0), row(.25)], context(True), True)
    support = c2.coverage([dict(target=target, weight=weight)], context(True))
    assert support[4] and support.sum() == 1
    assert not c2.coverage([dict(target=target[1:], weight=weight[1:])], context(True)).any()


def test_replay_filter_branch_histories_are_separate_and_clipping_is_last():
    a = np.full((5,45), .3, np.float32); a[1:3,4] = .95
    b = a.copy(); b[1:3,4] = .6
    times = np.arange(5) / 30
    expected = c2.replay(a, times, context(True))
    c2.replay(b, times, context(True))
    assert np.array_equal(expected, c2.replay(a, times, context(True)))
    assert np.isfinite(expected).all() and np.all((expected >= 0) & (expected <= 1))
    assert not np.array_equal(expected, c2.emit(a, context(True)))


def live_fixture(tmp_path):
    class Teacher(torch.nn.Module):
        def forward(self, stock, embedding):
            return torch.clamp(stock + embedding[:, :45] * .001, 0, 1)
    teacher_path = tmp_path / "reference.onnx"
    stock_path = tmp_path / "stock-fixture.bin"
    stock_path.write_bytes(b"live-feature fixture; no stock inference performed")
    torch.onnx.export(Teacher(), (torch.zeros(1,45), torch.zeros(1,1280)), str(teacher_path),
        input_names=["stock","embedding"], output_names=["personal"], opset_version=17, dynamo=False)
    graph = onnx.load(teacher_path)
    m = graph.metadata_props.add(); m.key="adapter_type"; m.value="embedding_head_v1"
    m = graph.metadata_props.add(); m.key="base_model_md5"; m.value=hashlib.md5(stock_path.read_bytes()).hexdigest()
    m = graph.metadata_props.add(); m.key="expression_schema_sha256"; m.value=hashlib.sha256("\n".join(EXPRESSION_NAMES).encode()).hexdigest()
    onnx.save(graph, teacher_path)
    contract = dict(Version=c2.VERSION, SchemaSha256=hashlib.sha256("\n".join(EXPRESSION_NAMES).encode()).hexdigest(),
        InputNormalization="gray_div255", ReferenceBlend=1, Output=context(True), EmbeddingDim=1280,
        **{k+s: str(teacher_path) if s=="Path" else c2.sha(teacher_path)
           for k in ("Stock","Feature","Reference") for s in ("Path","Sha256")})
    contract.update(StockPath=str(stock_path), StockSha256=c2.sha(stock_path))
    cp = tmp_path / "contract.json"; c2.write_new(cp, contract)
    sessions=[]
    rng=np.random.default_rng(40)
    for n, (task, value) in enumerate((("rest",0),("jaw-small",.25))):
        directory=tmp_path/task; (directory/"frames").mkdir(parents=True)
        entries=[]
        for i in range(12):
            entry=row(value); entry.update(i=i, t=i*333333, inference_timestamp=i*33+1,
                embedding_row=i, stock=np.full(45,.4).tolist())
            entries.append(entry)
            cv2.imwrite(str(directory/"frames"/f"{i:06}.jpg"), rng.integers(0,255,(224,224),dtype=np.uint8))
        (directory/"labels.jsonl").write_text("\n".join(json.dumps(r) for r in entries), encoding="utf-8")
        c2.write_new(directory/"session.json", dict(EndedUtc="fixture", FrameCount=12,
            C2=dict(ContractSha256=c2.sha(cp), StopwatchFrequency=1000)))
        rng.random((12,1280),dtype=np.float32).tofile(directory/"embeddings.f32")
        sessions.append(dict(Directory=str(directory), OriginId="one-wearing", Role="practice", TaskId=task, Legacy=False,
            SessionSha256=c2.sha(directory/"session.json"), LabelsSha256=c2.sha(directory/"labels.jsonl"), Review={"Accepted":True}))
    manifest=dict(Version=c2.VERSION, LabelRecipe=c2.RECIPE, ContractPath=str(cp), ContractSha256=c2.sha(cp), Sessions=sessions)
    mp=tmp_path/"manifest.json"; c2.write_new(mp,manifest)
    return mp,manifest,contract


def test_candidate_export_is_new_named_finite_and_never_claims_win(tmp_path):
    mp, manifest, contract=live_fixture(tmp_path)
    before=c2.sha(contract["ReferencePath"])
    run=c2.train(mp,tmp_path/"candidates",epochs=1)
    report=c2.read(run/"summary.json")
    assert report["recommended"] is False and report["status"]=="comparison_needed"
    assert report["development"]==[] and report["check"]==[]
    assert c2.sha(contract["ReferencePath"])==before
    runtime=c2.cpu_session(run/"candidate.onnx")
    assert {i.name for i in runtime.get_inputs()}=={"reference","embedding"}
    reference=np.full((1,45),.4,np.float32)
    result=runtime.run(["personal"],{"reference":reference,"embedding":np.ones((1,1280),np.float32)})[0]
    assert np.array_equal(result[:,5:],reference[:,5:])
    assert np.isfinite(result).all()


def test_source_changes_are_rejected(tmp_path):
    mp,m,c=live_fixture(tmp_path)
    Path(m["Sessions"][0]["Directory"],"labels.jsonl").write_text("{}",encoding="utf-8")
    out=tmp_path/"prepared";out.mkdir()
    with pytest.raises(ValueError,match="source changed"):
        c2.prepare(m,c,out)


def test_same_wearing_cannot_be_check_and_practice(tmp_path):
    mp,m,c=live_fixture(tmp_path)
    m["Sessions"][1]["Role"]="check"
    out=tmp_path/"prepared";out.mkdir()
    with pytest.raises(ValueError,match="share one wearing"):
        c2.prepare(m,c,out)
