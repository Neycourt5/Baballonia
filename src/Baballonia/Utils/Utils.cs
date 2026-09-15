using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security;
using System.Security.Cryptography;
using System.Security.Principal;

namespace Baballonia;

public static class Utils
{
    public const int EyeRawExpressions = 6;
    public const int FaceRawExpressions = 45;
    public const int FramesForEyeInference = 4;

    public static readonly bool IsSupportedDesktopOS = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() || OperatingSystem.IsLinux();

    public const int MobileWidth = 975;

    private const string k_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    // Timer resolution helpers
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Interoperability", "CA1401:PInvokesShouldNotBeVisible"), System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Security", "CA2118:ReviewSuppressUnmanagedCodeSecurityUsage"), SuppressUnmanagedCodeSecurity]
    [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod", SetLastError = true)]
    public static extern uint TimeBeginPeriod(uint uMilliseconds);

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Interoperability", "CA1401:PInvokesShouldNotBeVisible"), System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Security", "CA2118:ReviewSuppressUnmanagedCodeSecurityUsage"), SuppressUnmanagedCodeSecurity]
    [DllImport("winmm.dll", EntryPoint = "timeEndPeriod", SetLastError = true)]
    public static extern uint TimeEndPeriod(uint uMilliseconds);

    // Proc memory read helpers
    public const int ProcessVmRead = 0x0010;

    [DllImport("kernel32.dll")]
    public static extern IntPtr OpenProcess(int dwDesiredAccess, bool bInheritHandle, int dwProcessId);

    [DllImport("kernel32.dll")]
    public static extern bool ReadProcessMemory(int hProcess, IntPtr lpBaseAddress, byte[] lpBuffer, int dwSize, ref int lpNumberOfBytesRead);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool DeleteFile(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint GetFileAttributes(string lpFileName);

    public static readonly bool HasAdmin = OperatingSystem.IsWindows() && new WindowsPrincipal(WindowsIdentity.GetCurrent()).IsInRole(WindowsBuiltInRole.Administrator);

    /// <summary>
    /// An optional name that keeps this copy's settings and models to itself.
    /// </summary>
    /// <remarks>
    /// Set with the BABALLONIA_PROFILE environment variable. Empty - the default - is the ordinary
    /// installation and nothing changes, so an existing install keeps its settings exactly where it
    /// left them.
    ///
    /// It exists so two copies can run side by side doing different jobs: one driving eyes, another
    /// driving the face. Without it they share a settings file and each save overwrites the other's
    /// camera choices, which reads as settings randomly reverting rather than as a conflict.
    ///
    /// Read once here rather than wherever it is needed, because these directories are static
    /// readonly and must agree with each other for the whole run.
    /// </remarks>
    public static readonly string Profile =
        (Environment.GetEnvironmentVariable("BABALLONIA_PROFILE") ?? string.Empty).Trim();

    /// <summary>The data folder name, suffixed when a profile is named.</summary>
    private static readonly string DataFolderName =
        string.IsNullOrEmpty(Profile) ? "ProjectBabble" : $"ProjectBabble-{Profile}";

    // Development/tests can keep every profile-owned write in an isolated workspace. Requiring
    // a named profile prevents an accidental override of the ordinary installed app's data.
    private static readonly string? IsolatedDataRoot = string.IsNullOrEmpty(Profile) ? null :
        Environment.GetEnvironmentVariable("BABALLONIA_DATA_ROOT");

    public static readonly string UserAccessibleDataDirectory = Path.Combine(IsolatedDataRoot ?? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), DataFolderName);

    public static readonly string PersistentDataDirectory = IsSupportedDesktopOS
        ? Path.Combine(IsolatedDataRoot ?? Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), DataFolderName)
        : AppContext.BaseDirectory;

    public static readonly string ModelsDirectory = IsSupportedDesktopOS
        ? Path.Combine(PersistentDataDirectory, "Models")
        : AppContext.BaseDirectory;

    public static readonly string ModelDataDirectory = IsSupportedDesktopOS
        ? Path.Combine(PersistentDataDirectory, "ModelData")
        : AppContext.BaseDirectory;

    public static readonly string VrcftLibsDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "VRCFaceTracking",
        "CustomLibs");

    public static void ExtractEmbeddedResource(Assembly assembly, string resourceName, string file, bool overwrite = false)
    {
        // Extract the embedded model if it isn't already present
        if (File.Exists(file) && !overwrite) return;

        using var stm = assembly
            .GetManifestResourceStream(resourceName);

        using Stream outFile = File.Create(file);

        const int sz = 4096;
        var buf = new byte[sz];
        while (true)
        {
            if (stm == null) throw new FileNotFoundException(file);
            var nRead = stm.Read(buf, 0, sz);
            if (nRead < 1)
                break;
            outFile.Write(buf, 0, nRead);
        }
    }

    public static void OpenUrl(string URL)
    {
        try
        {
            Process.Start(URL);
        }
        catch
        {
            if (OperatingSystem.IsWindows())
            {
                var url = URL.Replace("&", "^&");
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }
            else if (OperatingSystem.IsMacOS())
            {
                Process.Start("open", URL);
            }
            else if (OperatingSystem.IsLinux())
            {
                Process.Start("xdg-open", URL);
            }
        }
    }

    public static string RandomString(int length = 6)
    {
        return new string(Enumerable.Repeat(k_chars, length).Select(s => s[Random.Shared.Next(s.Length)]).ToArray());
    }

    public static string GenerateMD5(string filepath)
    {
        // Credit to delta for this method https://github.com/XDelta/
        using var stream = File.OpenRead(filepath);
        using var md5 = MD5.Create();
        var hash = md5.ComputeHash(stream);
        return BitConverter.ToString(hash).Replace("-", "");
    }
}
