using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services.Inference;

/// <summary>A camera lifecycle observed by the watchdog and mutated only by its owning manager.</summary>
public interface IRecoverableCameraSlot
{
    string Name { get; }
    CameraTarget? Target { get; }
    CameraState State { get; }
    event Action<CameraState>? StateChanged;
    DateTime? SourceInstalledAtUtc { get; }
    TimeSpan? TimeSinceLastFrame { get; }
    bool IsTargetPresent(string address);
    Task<bool> RecoverAsync(TimeSpan firstFrameTimeout, CancellationToken cancellationToken);
}

public interface ICameraSlotHost
{
    IReadOnlyList<IRecoverableCameraSlot> Slots { get; }
}
