using System;
using OpenCvSharp;

namespace Baballonia.Services.Personalization;

/// <summary>
/// Cheap duplicate detection for pipeline frames.
/// </summary>
/// <remarks>
/// The processing tick runs at ~100 Hz while cameras deliver 30-60 fps, so the same Mat is re-served
/// several times in a row. Consumers that store frames need to skip those repeats, and they need to
/// do it on the tick, before any copying - which rules out hashing the whole image.
///
/// Duplicates here are bit-identical rather than merely similar, so a sparse sample separates them
/// as reliably as a full hash at a fraction of the cost. Shared by the dataset recorder and the
/// hard-example buffer so both agree on what "the same frame" means.
/// </remarks>
internal static class FrameChecksum
{
    private const int Samples = 64;
    private const ulong FnvOffsetBasis = 1469598103934665603UL;
    private const ulong FnvPrime = 1099511628211UL;

    /// <summary>FNV-1a over a fixed grid of pixels, with the byte count folded in.</summary>
    public static unsafe ulong Sparse(Mat mat)
    {
        if (mat is null || mat.Empty())
            return 0;

        var data = (byte*)mat.DataPointer;
        var total = (long)mat.Total() * mat.ElemSize();
        if (total <= 0 || data is null)
            return 0;

        var stride = Math.Max(1, total / Samples);

        var hash = FnvOffsetBasis;
        for (long offset = 0; offset < total; offset += stride)
        {
            hash ^= data[offset];
            hash *= FnvPrime;
        }

        // Fold in the length so a resolution change can never collide with the previous frame.
        hash ^= (ulong)total;
        return hash;
    }
}
