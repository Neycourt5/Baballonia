namespace Baballonia.Services.Inference;

public enum CameraState
{
    Stopped,
    Starting,
    Running,
    Reconnecting,
}

/// <summary>What the user wants a camera slot to run.</summary>
public sealed record CameraTarget(string Address, string Backend);
