using CommunityToolkit.Mvvm.ComponentModel;

namespace Baballonia.Models;

public partial class SliderBindableSetting : ObservableObject
{
    public string Name { get; set; }

    [ObservableProperty] private float _lower;
    [ObservableProperty] private float _currentExpression;

    [ObservableProperty] private float _upper;
    [ObservableProperty] private float _min;
    [ObservableProperty] private float _max;

    public SliderBindableSetting(string name, float lower = 0f, float upper = 1f, float min = 0f, float max = 1f)
    {
        Name = name;
        Lower = lower;
        Upper = upper;
        Min = min;
        Max = max;
    }
}
