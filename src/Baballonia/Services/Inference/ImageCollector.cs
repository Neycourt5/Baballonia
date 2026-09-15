using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Inference;

public class ImageCollector : IImageTransformer, System.IDisposable
{
    private const int QueueDepth = 5;
    private readonly Queue<Mat> _imageQueue = new();

    public void Reset()
    {
        while (_imageQueue.Count > 0)
            _imageQueue.Dequeue().Dispose();
    }

    public void Dispose() => Reset();

    public Mat? Apply(Mat image)
    {
        var split = image.Split();
        var merged = new Mat();
        try
        {
            foreach (var mat in split)
                Cv2.EqualizeHist(mat, mat);

            // Swap left and right because inference requires them in that order.
            Cv2.Merge(split.Reverse().ToArray(), merged);
        }
        catch
        {
            merged.Dispose();
            throw;
        }
        finally
        {
            // `merged` owns its own copy of the data.
            foreach (var mat in split)
                mat.Dispose();
        }

        _imageQueue.Enqueue(merged);

        if (_imageQueue.Count < QueueDepth)
            return null;

        var removed = _imageQueue.Dequeue();
        removed.Dispose();

        // feed the most recent matrix here at the start
        var last4 = _imageQueue.Skip(_imageQueue.Count - 4).Take(4).Reverse().ToArray();

        var channels = new List<Mat>();
        try
        {
            foreach (var m in last4)
                channels.AddRange(Cv2.Split(m));

            var octoMatrix = new Mat();
            try
            {
                Cv2.Merge(channels.ToArray(), octoMatrix);
                return octoMatrix;
            }
            catch
            {
                octoMatrix.Dispose();
                throw;
            }
        }
        finally
        {
            foreach (var channel in channels)
                channel.Dispose();
        }
    }
}
