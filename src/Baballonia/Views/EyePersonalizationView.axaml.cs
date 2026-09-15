using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace Baballonia.Views;

public partial class EyePersonalizationView : UserControl
{
    public EyePersonalizationView()
    {
        InitializeComponent();
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }
}
