namespace BabblePersonalizer.Core.Storage;

public sealed class PersonalizerDataPaths
{
    public string Root { get; }
    public string Sessions => Path.Combine(Root, "Sessions");
    public string Profiles => Path.Combine(Root, "Profiles");
    public string Exports => Path.Combine(Root, "Exports");
    public string Reports => Path.Combine(Root, "Reports");
    public string Backups => Path.Combine(Root, "Backups");

    public PersonalizerDataPaths(string? root = null)
    {
        Root = root ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "BabblePersonalizerData");
    }

    public void EnsureCreated()
    {
        foreach (var path in new[] { Root, Sessions, Profiles, Exports, Reports, Backups }) Directory.CreateDirectory(path);
    }
}
