using System;
using System.Linq;
using System.Reflection;
using Baballonia.Desktop.Calibration;
using Baballonia.Services.Calibration;
using Baballonia.Services.Personalization;
using Baballonia.Services.Personalization.C2;
using Baballonia.ViewModels.SplitViewPane;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

[TestClass]
public class C2HeadsetInstructionTest
{
    // Exercise the exact production handoff without constructing a ViewModel that starts a UI
    // timer and inventories a profile. All labels and presenter events here are synthetic.
    private delegate bool PublishCue(CueStateSource cue, CuePhase phase,
        IVrCalibrationPresenter? presenter, VrCalibrationFrame frame, out bool cancelled);

    private static readonly Func<C2Task, string, double, VrCalibrationFrame> BuildFrame =
        typeof(C2ViewModel).GetMethod("BuildHeadsetFrame", BindingFlags.NonPublic | BindingFlags.Static)!
            .CreateDelegate<Func<C2Task, string, double, VrCalibrationFrame>>();
    private static readonly PublishCue Publish =
        typeof(C2ViewModel).GetMethod("TryPublishCaptureCue", BindingFlags.NonPublic | BindingFlags.Static)!
            .CreateDelegate<PublishCue>();

    private static C2Task Task => C2Task.All.Single(task => task.Id == "jaw-small");

    [TestMethod]
    public void HoldInstructionStaysStableWhileProgressChanges()
    {
        var first = BuildFrame(Task, "hold", 4);
        var next = BuildFrame(Task, "hold", 4.2);

        Assert.AreEqual(Task.Instruction, first.Instruction);
        Assert.AreEqual(first.Instruction, next.Instruction,
            "Saved-frame counts and status text must not turn each tick into a new instruction.");
        Assert.AreNotEqual(first.PhaseProgress, next.PhaseProgress);
        Assert.IsTrue(OpenVrCalibrationPresenter.IsCosmeticUpdate(next, first),
            "The renderer must be able to throttle or retry progress-only C2 updates.");
    }

    [TestMethod]
    public void ANewTaskInstructionStillRequiresAnImmediateUpdate()
    {
        var first = BuildFrame(Task, "hold", 4);
        var next = BuildFrame(Task with { Instruction = "Relax your jaw now." }, "hold", 4.2);

        Assert.IsFalse(OpenVrCalibrationPresenter.IsCosmeticUpdate(next, first));
    }

    [TestMethod]
    [DataRow("prepare", 1.5, VrCalibrationPhase.Preparing, 2.0)]
    [DataRow("settling", 3.375, VrCalibrationPhase.Settling, 5.0)]
    [DataRow("hold", 5.75, VrCalibrationPhase.Hold, 2.0)]
    public void FrameUsesStructuredPhaseCountdownAndProgress(
        string phase, double elapsed, VrCalibrationPhase expectedPhase, double countdown)
    {
        var frame = BuildFrame(Task, phase, elapsed);

        Assert.AreEqual(expectedPhase, frame.Phase);
        Assert.AreEqual(0.5, frame.PhaseProgress, 1e-9);
        Assert.AreEqual(countdown, frame.CountdownSeconds);
        Assert.IsTrue(frame.OverallProgress > 0 && frame.OverallProgress < 1);
        Assert.AreEqual(Task.Instruction, frame.Instruction);
        Assert.IsTrue(frame.AllowCancel);
    }

    [TestMethod]
    public void CaptureProgressIsClampedAtTheEnds()
    {
        var before = BuildFrame(Task, "prepare", -0.1);
        var after = BuildFrame(Task, "hold", 10);

        Assert.AreEqual(0d, before.OverallProgress);
        Assert.AreEqual(0d, before.PhaseProgress);
        Assert.AreEqual(1d, after.OverallProgress);
        Assert.AreEqual(1d, after.PhaseProgress);
    }

    [TestMethod]
    public void PhaseChangePublishesTheCueOnlyAfterPresentationAndInputAreHealthy()
    {
        var cue = Cue();
        cue.SetPhase(Phase("settling"));
        var presenter = new ObservingPresenter(cue);

        Assert.IsTrue(Publish(cue, Phase("hold"), presenter, BuildFrame(Task, "hold", 4), out var cancelled));

        Assert.IsFalse(cancelled);
        Assert.IsNull(presenter.CueDuringPresent,
            "Camera frames during the instruction transition must not receive either phase label.");
        Assert.IsNull(presenter.CueDuringPoll,
            "The new cue must wait until controller input has also been checked.");
        Assert.AreEqual("hold", cue.CurrentCue()!.Phase);
        Assert.AreEqual(Task.Id, cue.CurrentCue()!.Id);
        Assert.AreEqual(Task.Target()[4], cue.CurrentCue()!.Target[4]);
        Assert.AreEqual(1, presenter.PresentCalls);
        Assert.AreEqual(1, presenter.PollCalls);
    }

    [TestMethod]
    public void CosmeticTicksKeepTheExistingValidCueDuringUpload()
    {
        var cue = Cue();
        cue.SetPhase(Phase("hold"));
        var presenter = new ObservingPresenter(cue);

        Assert.IsTrue(Publish(cue, Phase("hold"), presenter, BuildFrame(Task, "hold", 4.2), out _));
        Assert.AreEqual("hold", presenter.CueDuringPresent?.Phase,
            "A progress update must not create recurring gaps in otherwise valid capture labels.");
        Assert.AreEqual("hold", presenter.CueDuringPoll?.Phase);
    }

    [TestMethod]
    [DataRow("unhealthy-before")]
    [DataRow("closed-before")]
    [DataRow("unhealthy-present")]
    [DataRow("closed-present")]
    [DataRow("throw-present")]
    [DataRow("unhealthy-poll")]
    [DataRow("throw-poll")]
    public void PresenterFailureClearsCueBeforeTheCallerFinalizesRecording(string failure)
    {
        var cue = Cue();
        cue.SetPhase(Phase("hold"));
        var presenter = new ObservingPresenter(cue) { Failure = failure };

        Assert.IsFalse(Publish(cue, Phase("hold"), presenter, BuildFrame(Task, "hold", 4.2), out var cancelled));

        Assert.IsFalse(cancelled);
        Assert.IsNull(cue.CurrentCue(),
            "A camera frame arriving before asynchronous FinishAsync must already be unlabelled.");
        if (failure.EndsWith("before", StringComparison.Ordinal))
            Assert.AreEqual(0, presenter.PresentCalls);
        if (failure.EndsWith("present", StringComparison.Ordinal))
            Assert.AreEqual(0, presenter.PollCalls);
    }

    [TestMethod]
    public void ControllerCancelDoesNotPublishThePendingPhase()
    {
        var cue = Cue();
        cue.SetPhase(Phase("settling"));
        var presenter = new ObservingPresenter(cue) { NextAction = VrCalibrationAction.Cancel };

        Assert.IsFalse(Publish(cue, Phase("hold"), presenter, BuildFrame(Task, "hold", 4), out var cancelled));
        Assert.IsTrue(cancelled);
        Assert.IsNull(cue.CurrentCue());
    }

    [TestMethod]
    public void DesktopCaptureStillPublishesItsCueWithoutAHeadset()
    {
        var cue = Cue();

        Assert.IsTrue(Publish(cue, Phase("hold"), null, BuildFrame(Task, "hold", 4), out var cancelled));
        Assert.IsFalse(cancelled);
        Assert.AreEqual("hold", cue.CurrentCue()!.Phase);
    }

    private static CueStateSource Cue() => new(() => 1_000, 1_000) { Source = "c2_instruction" };

    private static CuePhase Phase(string phase) => new(Task.Id, phase, Task.Dims,
        Task.Target(), Task.Target(), 1, 1_000);

    private sealed class ObservingPresenter(CueStateSource cue) : IVrCalibrationPresenter
    {
        public string Failure { get; init; } = "";
        public VrCalibrationAction NextAction { get; init; }
        public int PresentCalls { get; private set; }
        public int PollCalls { get; private set; }
        public FrameLabel.CueLabel? CueDuringPresent { get; private set; }
        public FrameLabel.CueLabel? CueDuringPoll { get; private set; }
        public bool IsAvailable => true;
        public bool IsPresenting => Failure != "closed-before" &&
                                    !(Failure == "closed-present" && PresentCalls > 0);
        public bool IsHealthy => Failure != "unhealthy-before" &&
                                 !(Failure == "unhealthy-present" && PresentCalls > 0) &&
                                 !(Failure == "unhealthy-poll" && PollCalls > 0);
        public string Status => "Synthetic presenter";
        public VrPresenterStartResult Begin(string sessionTitle) => VrPresenterStartResult.Success();
        public void Present(VrCalibrationFrame frame)
        {
            PresentCalls++;
            CueDuringPresent = cue.CurrentCue();
            if (Failure == "throw-present") throw new InvalidOperationException("Synthetic upload failure");
        }
        public VrCalibrationAction ConsumeAction()
        {
            PollCalls++;
            CueDuringPoll = cue.CurrentCue();
            if (Failure == "throw-poll") throw new InvalidOperationException("Synthetic input failure");
            return NextAction;
        }
        public void End(string? completionMessage = null) { }
        public void Dispose() { }
    }
}
