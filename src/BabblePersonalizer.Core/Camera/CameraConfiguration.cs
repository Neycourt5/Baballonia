namespace BabblePersonalizer.Core.Camera;

public sealed record CropRegion(int X = 0, int Y = 0, int Width = 256, int Height = 256);

public sealed record CameraConfiguration(
    int DeviceIndex = 0,
    int Width = 640,
    int Height = 480,
    double FramesPerSecond = 30,
    CropRegion? Crop = null,
    double RotationRadians = 0,
    double Gamma = 1,
    bool UseRedChannel = false,
    bool HorizontalMirror = false,
    bool VerticalMirror = false)
{
    public CropRegion EffectiveCrop => Crop ?? new CropRegion(0, 0, Width, Height);
}

public sealed record CameraDevice(int Index, string DisplayName);
