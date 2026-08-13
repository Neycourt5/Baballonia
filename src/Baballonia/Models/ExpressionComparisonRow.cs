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
