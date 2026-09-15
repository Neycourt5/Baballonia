using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services;
using Baballonia.Services.Calibration;
using Baballonia.Services.events;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Filters;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.C2;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using OpenCvSharp;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class C2Test
{
    [TestMethod]
    public void RuntimeEnforcesPreservedChannelsEvenWhenTheGraphWouldChangeThem()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Assets", "PersonalModels", "c2Adapter.onnx");
        using var candidate = new C2CandidateCorrector(path, "fixture-context", supportedChannels: new System.Collections.Generic.HashSet<int> {19, 20});
        var reference = Enumerable.Repeat(.3f, 45).ToArray();
        reference[33] = 1.2f;
        var embedding = new DenseTensor<float>([1, 1280]);
        foreach (var amount in new[] { 1f, .5f })
        {
            candidate.Contribution = amount;
            var result = candidate.Correct(reference, embedding);
            Assert.IsFalse(candidate.CorrectsChannel(4));
            Assert.AreEqual(reference[4], result[4], "Fixture changes JawOpen, but the declared preservation boundary must win.");
            Assert.AreEqual(reference[33], result[33], "Uncorrected channels preserve even out-of-range reference values.");
            Assert.IsNull(candidate.Failure);
        }
    }

    [TestMethod]
    public void RuntimeSelectsNamedOutputAndFallsBackToThePersonalReference()
    {
        var path=Path.Combine(AppContext.BaseDirectory,"Assets","PersonalModels","c2Adapter.onnx");
        using var candidate=new C2CandidateCorrector(path,"fixture-context");
        var reference=Enumerable.Repeat(.3f,45).ToArray();
        var embedding=new DenseTensor<float>([1,1280]);
        Assert.AreEqual(.4f,candidate.Correct(reference,embedding)[4],1e-6f);
        candidate.Contribution=0;
        CollectionAssert.AreEqual(reference,candidate.Correct(reference,embedding));
        candidate.Contribution=.5f;
        Assert.AreEqual(.35f,candidate.Correct(reference,embedding)[4],1e-6f);
        candidate.Correct(reference,null);
        Assert.IsNotNull(candidate.Failure);
        CollectionAssert.AreEqual(reference,candidate.Correct(reference,embedding));
        Assert.AreEqual(.3f,reference[4]);
    }

    [TestMethod]
    public void RuntimeRejectsChangedContextWithoutLoadingItAsLegacyC()
    {
        var path=Path.Combine(AppContext.BaseDirectory,"Assets","PersonalModels","c2Adapter.onnx");
        Assert.ThrowsExactly<InvalidDataException>(()=>new C2CandidateCorrector(path,"different-context"));
    }

    private static C2OutputContext Context(bool nonIdentity = false) => new(
        Enumerable.Range(0,45).Select(_ => nonIdentity ? new[] {.2f,.8f,0f,1f} : new[] {0f,1f,0f,1f}).ToArray(),
        1, true, .5f, 3);

    [TestMethod]
    public void TargetsInvertOnlyTheActualPresentationMap()
    {
        var context = Context(true);
        Assert.AreEqual(.35f, context.ToRawTarget(4,.25f), 1e-6f);
        Assert.AreEqual(.25f, context.Emit(4,context.ToRawTarget(4,.25f)), 1e-6f);
        Assert.AreEqual(0f, context.Emit(4,-.1f));
        Assert.AreEqual(1f, context.Emit(4,1.1f));
    }

    [TestMethod]
    public void TasksDoNotEquateTeethOrClosedLipsWithJawOpening()
    {
        Assert.AreEqual(0, C2Task.All.Single(t => t.Id == "toothy").Dims.Length);
        Assert.AreEqual(0, C2Task.All.Single(t => t.Id == "speech").Dims.Length);
        CollectionAssert.AreEqual(new[] {19,20}, C2Task.All.Single(t => t.Id == "smile-natural").Dims);
        CollectionAssert.AreEqual(new[] {4,19,20}, C2Task.All.Single(t => t.Id == "rest").Dims);
    }

    [TestMethod]
    public void WearingIdentitySurvivesRestartAndOnlyExplicitReseatChangesIt()
    {
        var root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        var first = new C2Workspace(root, Path.Combine(root,"legacy"));
        var id = first.GetOrCreateOrigin();
        var reopened = new C2Workspace(root, Path.Combine(root,"legacy"));
        Assert.AreEqual(id, reopened.GetOrCreateOrigin());
        Assert.AreNotEqual(id, reopened.GetOrCreateOrigin(true));
    }

    [TestMethod]
    public void OldNeutralIsNotSilentlyRelabelledAndReviewDoesNotModifySource()
    {
        var root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        var directory = Path.Combine(root,"legacy","one"); Directory.CreateDirectory(directory);
        var m = new SessionMetadata { SessionId="one",SessionType="Neutral",EndedUtc="fixture",FrameCount=1,ImageWidth=224,ImageHeight=224 };
        var path = Path.Combine(directory,"session.json"); C2Contract.WriteAtomic(path,m);
        File.WriteAllText(Path.Combine(directory,"labels.jsonl"), "{}");
        var before = C2Contract.Hash(path);
        var workspace = new C2Workspace(Path.Combine(root,"C2"), Path.Combine(root,"legacy"));
        var row = workspace.Inventory().Single();
        Assert.AreEqual("Needs review",row.State);
        StringAssert.Contains(row.Reason,"does not verify 45 zero");
        workspace.Review(row,true,"replay only");
        Assert.AreEqual("Replay / preparation",workspace.Inventory().Single().State);
        Assert.AreEqual(before,C2Contract.Hash(path));
        File.AppendAllText(Path.Combine(directory,"labels.jsonl"), "\n{}");
        Assert.AreEqual("Needs review",workspace.Inventory().Single().State,
            "A source mutation must not silently inherit an earlier acceptance.");
    }

    [TestMethod]
    public async Task C2RecorderCopiesSameFrameFeaturesAndDrainsBeforeReleasingGate()
    {
        var bus = new FacePipelineEventBus();
        var gate = new TrainingCaptureGate();
        using var recorder = new DatasetRecorderService(bus,NullLogger<DatasetRecorderService>.Instance,captureGate:gate);
        var root = Path.Combine(Path.GetTempPath(),Guid.NewGuid().ToString("N"));
        var context = new C2CaptureContext("origin","practice","rest","fixture","hash",1280,1000);
        var id = recorder.StartSession(SessionType.Guided,datasetRoot:root,c2:context);
        Assert.IsTrue(File.Exists(Path.Combine(root,id,"session.json")));
        using var mat = new Mat(224,224,MatType.CV_8UC1,new Scalar(50));
        var feature = new DenseTensor<float>([1,1280]); feature[0,0]=17;
        var stock = Enumerable.Repeat(.3f,45).ToArray();
        bus.Publish(new FacePipelineEvents.NewRawExpressionsEvent(mat,stock,100,feature,30));
        feature[0,0]=999; stock[4]=999;
        var summary = await recorder.StopSessionAsync();
        Assert.AreEqual(1,summary!.FrameCount);
        using var binary = new BinaryReader(File.OpenRead(Path.Combine(summary.Directory,"embeddings.f32")));
        Assert.AreEqual(17f,binary.ReadSingle());
        var row = JsonSerializer.Deserialize<FrameLabel>(File.ReadAllLines(Path.Combine(summary.Directory,"labels.jsonl"))[0])!;
        Assert.AreEqual(.3f,row.Stock[4]); Assert.AreEqual(30,row.InferenceTimestamp);
        Assert.AreEqual(0,row.EmbeddingRow); Assert.IsNull(gate.ActiveActivity);
    }

    [TestMethod]
    public void FilterReplayUsesIndependentHistoriesAndKeepsBlendEndpoints()
    {
        var now = new DateTime(2026,1,1,0,0,0,DateTimeKind.Utc);
        var baselineFilter = new OneEuroFilter(.5f,3,clock:()=>now);
        var zeroFilter = new OneEuroFilter(.5f,3,clock:()=>now);
        var fullFilter = new OneEuroFilter(.5f,3,clock:()=>now);
        var candidateFilter = new OneEuroFilter(.5f,3,clock:()=>now);
        var samples = new List<float[]>(); var emitted = new List<float[]>(); var times = new List<double>();
        foreach (var raw in new[] {.3f,.95f,.95f,.3f,.3f})
        {
            var reference = Enumerable.Repeat(.3f,45).ToArray(); reference[4]=raw;
            var candidate = (float[])reference.Clone(); candidate[4]=Math.Min(raw,.6f);
            OrderedFloatMap Map(float[] v) { var m=new OrderedFloatMap(PersonalizationSchemaBinding.ExpectedKeys.ToArray()); v.CopyTo(m.ValuesSpan); return m; }
            var b=baselineFilter.Filter(Map(reference)).ValuesSpan.ToArray();
            CollectionAssert.AreEqual(b,zeroFilter.Filter(Map(reference)).ValuesSpan.ToArray());
            CollectionAssert.AreEqual(candidateFilter.Filter(Map(candidate)).ValuesSpan.ToArray(),fullFilter.Filter(Map(candidate)).ValuesSpan.ToArray());
            samples.Add(reference); emitted.Add(b.Select((v,i)=>Context(true).Emit(i,v)).ToArray());
            times.Add((now-new DateTime(2026,1,1,0,0,0,DateTimeKind.Utc)).TotalSeconds);
            now=now.AddMilliseconds(33);
        }
        var output=Environment.GetEnvironmentVariable("C2_REPLAY_FIXTURE");
        if (!string.IsNullOrWhiteSpace(output)) C2Contract.WriteAtomic(output,new {samples,emitted,times,context=Context(true)});
        Assert.AreNotEqual(Context(true).Emit(4,.95f),emitted[1][4]);
    }

    [TestMethod]
    public void C2RawContractMatchesActualSenderWithNonIdentityRangesAndCurrentJawKey()
    {
        var calibration=new Mock<ICalibrationService>();
        calibration.Setup(c=>c.GetExpressionSettings(It.IsAny<string>())).Returns(new CalibrationParameter(.2f,.8f,0,1));
        var loop=(ProcessingLoopService)RuntimeHelpers.GetUninitializedObject(typeof(ProcessingLoopService));
        var sender=new ParameterSenderService(null!,null!,null!,calibration.Object,loop,NullLogger<ParameterSenderService>.Instance);
        typeof(ParameterSenderService).GetField("_jawOpenCurve",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(sender,2f);
        var map=new OrderedFloatMap(PersonalizationSchemaBinding.ExpectedKeys.ToArray()); map.ValuesSpan.Fill(.35f);
        typeof(ParameterSenderService).GetMethod("ProcessFaceExpressionData",BindingFlags.Instance|BindingFlags.NonPublic)!.Invoke(sender,[map]);
        var queue=(System.Collections.Concurrent.ConcurrentQueue<OscCore.OscMessage>)typeof(ParameterSenderService)
            .GetField("_vrcftQueue",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(sender)!;
        Assert.AreEqual(45,queue.Count);
        Assert.AreEqual(Context(true).Emit(4,.35f),(float)queue.Single(m=>m.Address.EndsWith("/jawOpen"))[0],1e-6);
    }
}
