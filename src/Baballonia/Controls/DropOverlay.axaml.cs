using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Baballonia.Contracts;
using Baballonia.Services;
using CommunityToolkit.Mvvm.DependencyInjection;

namespace Baballonia.Controls;

public partial class DropOverlay : UserControl
{
    private readonly ILocalSettingsService _localSettingsService;

    public DropOverlay()
    {
        _localSettingsService = Ioc.Default.GetRequiredService<ILocalSettingsService>();
        InitializeComponent(true);
        IsOverlayVisible = DataUploadConfiguration.FromEnvironment().IsConfigured;
    }

    public static readonly StyledProperty<bool> IsOverlayVisibleProperty =
        AvaloniaProperty.Register<DropOverlay, bool>(nameof(IsOverlayVisible), false);

    public bool IsOverlayVisible
    {
        get => GetValue(IsOverlayVisibleProperty);
        set => SetValue(IsOverlayVisibleProperty, value);
    }

    private void Changed(object? sender, RoutedEventArgs e)
    {
        if (DataUploadConfiguration.FromEnvironment().IsConfigured)
            _localSettingsService.SaveSetting("AppSettings_ShareEyeData", EyeDataCheckbox.IsChecked);
    }

    private void EyeDataOptInRead(object? sender, RoutedEventArgs e)
    {
        _localSettingsService.SaveSetting("EyeDataOptInRead", true);
        IsOverlayVisible = false;
    }
}
