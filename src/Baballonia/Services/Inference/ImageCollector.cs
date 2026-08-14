using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;

namespace Baballonia.Services.Inference;

public class ImageCollector : IImageTransformer, System.IDisposable
{
    /// <summary>Frames kept so the model can see motion; the newest four are fed to it.</summary>
    private const int QueueDepth = 5;

    private Queue<Mat> ImageQueue = new();

    /// <summary>
    /// Drops the frame history.
    /// </summary>
    /// <remarks>
    /// Needed whenever the camera changes: the queue would otherwise splice frames from the old
    /// camera into the temporal stack, so the model would see four "consecutive" frames that are
    /// nothing of the sort. Also releases the retained Mats rather than waiting for a finalizer.
    /// </remarks>
    public void Reset()
    {
        while (ImageQueue.Count > 0)
            ImageQueue.Dequeue().Dispose();
    }

    public void Dispose() => Reset();

    public Mat? Apply(Mat image)
    {
        Mat[] split = image.Split();
        foreach (var mat in split)
        {
            Cv2.EqualizeHist(mat, mat);
        }

        Mat merged = new Mat();
        // swap left and right because inference requires them in that way
        Cv2.Merge(split.Reverse().ToArray(), merged);

        // Split() hands out new Mats; the merge copied what it needed, so holding them any longer
        // leaks two native buffers on every tick.
        foreach (var mat in split)
            mat.Dispose();

        ImageQueue.Enqueue(merged);

        if (ImageQueue.Count < QueueDepth)
            return null;

        var removed = ImageQueue.Dequeue();
        removed.Dispose();

        // feed the most recent matrix here at the start
        var last4 = ImageQueue.Skip(ImageQueue.Count - 4).Take(4).Reverse().ToArray();

        var channels = new List<Mat>();
        foreach (var m in last4)
        {
            Mat[] splitChannels = Cv2.Split(m);
            channels.AddRange(splitChannels);
        }

        Mat octoMatrix = new Mat();
        Cv2.Merge(channels.ToArray(), octoMatrix);

        foreach (var channel in channels)
            channel.Dispose();

        // Freshly allocated and not retained here - the caller owns it and must dispose it.
        return octoMatrix;
    }
}
