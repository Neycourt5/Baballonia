using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Baballonia.Services.Personalization;
using Baballonia.ViewModels.SplitViewPane;

namespace Baballonia.Views;

public partial class PersonalizationView : UserControl
{
    private static readonly FilePickerFileType OnnxModels = new("ONNX personal models")
    {
        Patterns = ["*.onnx"],
    };

    public PersonalizationView()
    {
        InitializeComponent();
    }

    private async void BrowsePersonalModel(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not PersonalizationViewModel viewModel) return;
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage == null) return;

        System.IO.Directory.CreateDirectory(PersonalizationPaths.ModelsRoot);
        var start = await storage.TryGetFolderFromPathAsync(PersonalizationPaths.ModelsRoot);
        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select a compatible personal face model",
            AllowMultiple = false,
            SuggestedStartLocation = start,
            FileTypeFilter = [OnnxModels],
        });
        if (files.Count == 0) return;

        await viewModel.SelectManualModelAsync(files[0].Path.LocalPath);
    }
}
