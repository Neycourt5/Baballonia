using BabblePersonalizer.Core.Camera;
using OpenCvSharp;

namespace BabblePersonalizer.Tests;

[TestClass]
public sealed class PreprocessorTests
{
    [TestMethod]
    public void TransformProducesExactRequestedGrayInputShape()
    {
        using var source = new Mat(100, 120, MatType.CV_8UC3, new Scalar(10, 20, 30));
        var service = new BaballoniaCompatiblePreprocessor();
        var config = new CameraConfiguration(Crop: new CropRegion(10, 10, 80, 60));
        using var transformed = service.Transform(source, config, 224, 224);
        Assert.AreEqual(224, transformed.Width);
        Assert.AreEqual(224, transformed.Height);
        Assert.AreEqual(1, transformed.Channels());
        var tensor = service.ToTensor(transformed);
        CollectionAssert.AreEqual(new[] { 1, 1, 224, 224 }, tensor.Dimensions.ToArray());
        Assert.IsTrue(tensor.ToArray().All(x => x is >= 0 and <= 1));
    }

    [TestMethod]
    public void InvalidCropFallsBackToWholeFrame()
    {
        using var source = new Mat(16, 16, MatType.CV_8UC1, Scalar.All(127));
        var service = new BaballoniaCompatiblePreprocessor();
        using var transformed = service.Transform(source,
            new CameraConfiguration(Crop: new CropRegion(100, 100, 20, 20)), 8, 8);
        Assert.AreEqual(8, transformed.Width);
        Assert.AreEqual(8, transformed.Height);
        Assert.AreEqual(127, transformed.At<byte>(0, 0));
    }
}
