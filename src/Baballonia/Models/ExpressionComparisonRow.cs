using System;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Baballonia.Models;

/// <summary>
/// One row of the stock-vs-personal debug view.
///
/// Rows are created once and mutated in place. Rebuilding the collection would churn the UI at the
/// update rate and make the list flicker; the whole point of this view is to watch values move.
/// </summary>
public partial class ExpressionComparisonRow(int index, string name) : ObservableObject
{
    public int Index { get; } = index;
    public string Name { get; } = name;

    [ObservableProperty] private float _stock;
    [ObservableProperty] private float _personal;
    [ObservableProperty] private float _delta;

    /// <summary>Absolute delta, used to surface the expressions the model is changing most.</summary>
    public float AbsoluteDelta => Delta < 0 ? -Delta : Delta;

    public void Update(float stock, float personal)
    {
        Stock = stock;
        Personal = personal;
        Delta = personal - stock;
        OnPropertyChanged(nameof(AbsoluteDelta));
    }
}

/// <summary>
/// Thread-friendly snapshots behind the stock-vs-personal table. Raw stock values always advance;
/// when no personal corrector is active, the personal side mirrors that same frame exactly.
/// </summary>
public sealed class ExpressionComparisonBuffer
{
    private readonly float[] _stock;
    private readonly float[] _personal;

    public ExpressionComparisonBuffer(int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        _stock = new float[count];
        _personal = new float[count];
    }

    public int Count => _stock.Length;
    public float StockAt(int index) => _stock[index];
    public float PersonalAt(int index) => _personal[index];

    public void AcceptRaw(float[] raw, bool personalCorrectorActive)
    {
        ArgumentNullException.ThrowIfNull(raw);
        CopySnapshot(raw, _stock);

        // A stock-only pipeline emits no corrected event. Mirroring here keeps both columns live
        // after switching away from a personal model instead of freezing the last corrected frame.
        if (!personalCorrectorActive)
            CopySnapshot(raw, _personal);
    }

    public void AcceptCorrected(float[] raw, float[] corrected)
    {
        ArgumentNullException.ThrowIfNull(raw);
        ArgumentNullException.ThrowIfNull(corrected);
        CopySnapshot(raw, _stock);
        CopySnapshot(corrected, _personal);
    }

    private static void CopySnapshot(float[] source, float[] destination)
    {
        var count = Math.Min(source.Length, destination.Length);
        Array.Copy(source, destination, count);
        if (count < destination.Length)
            Array.Clear(destination, count, destination.Length - count);
    }
}
