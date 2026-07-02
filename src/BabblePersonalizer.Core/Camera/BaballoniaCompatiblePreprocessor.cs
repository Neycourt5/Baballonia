using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace BabblePersonalizer.Core.Camera;

/// <summary>Standalone reproduction of Baballonia's face ImageTransformer + MatToFloatTensorConverter.</summary>
public sealed class BaballoniaCompatiblePreprocessor
{
    public Mat Transform(Mat image, CameraConfiguration configuration, int targetWidth, int targetHeight)
    {
        var crop = configuration.EffectiveCrop;
        var valid = crop.X >= 0 && crop.Y >= 0 && crop.Width > 0 && crop.Height > 0 &&
                    crop.X + crop.Width <= image.Width && crop.Y + crop.Height <= image.Height &&
                    crop.Width != image.Width && crop.Height != image.Height;
        var roi = valid ? new Rect(crop.X, crop.Y, crop.Width, crop.Height) : new Rect(0, 0, image.Width, image.Height);
        using var roiMat = new Mat(image, roi);
        var result = roiMat.Clone();
        if (result.Channels() >= 2)
        {
            var gray = new Mat();
            if (configuration.UseRedChannel) Cv2.ExtractChannel(result, gray, 0); // Matches Baballonia exactly.
            else Cv2.CvtColor(result, gray, ColorConversionCodes.BGR2GRAY);
            result.Dispose(); result = gray;
        }
        if (Math.Abs(configuration.Gamma - 1) > double.Epsilon)
        {
            var adjusted = new Mat(); result.ConvertTo(adjusted, result.Type(), configuration.Gamma);
            result.Dispose(); result = adjusted;
        }
        var target = new Size(targetWidth, targetHeight);
        if (Math.Abs(configuration.RotationRadians) > double.Epsilon || configuration.HorizontalMirror || configuration.VerticalMirror)
        {
            var cos = Math.Cos(configuration.RotationRadians); var sin = Math.Sin(configuration.RotationRadians);
            var scale = 1 / (Math.Abs(cos) + Math.Abs(sin));
            var hscale = (configuration.HorizontalMirror ? -1 : 1) * scale;
            var vscale = (configuration.VerticalMirror ? -1 : 1) * scale;
            using var matrix = new Mat<double>(2, 3); var data = matrix.AsSpan<double>();
            data[0] = (double)targetWidth / result.Width * cos * hscale;
            data[1] = (double)targetHeight / result.Height * sin * hscale;
            data[2] = (targetWidth - (targetWidth * cos + targetHeight * sin) * hscale) * .5;
            data[3] = -(double)targetWidth / result.Width * sin * vscale;
            data[4] = (double)targetHeight / result.Height * cos * vscale;
            data[5] = (targetHeight + (targetWidth * sin - targetHeight * cos) * vscale) * .5;
            Cv2.WarpAffine(result, result, matrix, target);
        }
        else Cv2.Resize(result, result, target);
        return result;
    }

    public DenseTensor<float> ToTensor(Mat transformed)
    {
        using var floatMat = new Mat();
        transformed.ConvertTo(floatMat, MatType.CV_32FC(transformed.Channels()), 1f / 255f);
        var channels = floatMat.Channels();
        var tensor = new DenseTensor<float>(new[] { 1, channels, floatMat.Rows, floatMat.Cols });
        for (var y = 0; y < floatMat.Rows; y++)
        for (var x = 0; x < floatMat.Cols; x++)
        for (var c = 0; c < channels; c++)
            tensor[0, c, y, x] = floatMat.At<float>(y, x * channels + c);
        return tensor;
    }
}
