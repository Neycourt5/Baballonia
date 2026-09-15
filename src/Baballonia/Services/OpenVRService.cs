using Microsoft.Extensions.Logging;
using System;
using System.IO;
using Valve.VR;

namespace Baballonia.Services;

public class OpenVRService(ILogger<OpenVRService> logger)
{
    //app key needed for vrmanifest
    private const string ApplicationKey = "projectbabble.Baballonia";

    private readonly object _sessionLock = new();

    /// <summary>
    /// True while a long-lived OpenVR context is held on behalf of a headset overlay.
    /// </summary>
    /// <remarks>
    /// The short-lived <see cref="WithSession"/> pattern below deliberately calls
    /// <c>OpenVR.Shutdown</c>, which invalidates <em>every</em> cached native interface pointer in
    /// the process - including an overlay handle a calibration presenter is actively drawing with.
    /// So while an overlay is out, autostart calls borrow that context and leave it running.
    /// </remarks>
    private bool _overlayContextHeld;

    private string _runtimeStatus = "SteamVR has not been checked yet.";

    /// <summary>Last thing OpenVR told us, for the calibration UI's status line.</summary>
    public string RuntimeStatus { get { lock (_sessionLock) return _runtimeStatus; } }

    /// <summary>
    /// Initializes (once) the process-wide OpenVR background context and returns its overlay API,
    /// or null with a human-readable <paramref name="status"/> explaining why it could not.
    /// </summary>
    /// <remarks>
    /// Guided calibration presents its instructions inside the headset rather than mirroring a
    /// desktop window, which needs an overlay interface that outlives a single call. The context is
    /// intentionally never shut down afterwards: tearing it down under a live overlay is what used
    /// to make the headset overlay flicker and then vanish.
    /// </remarks>
    public CVROverlay? TryGetOverlay(out string status)
    {
        lock (_sessionLock)
        {
            if (!_overlayContextHeld)
            {
                try
                {
                    if (!OpenVR.IsRuntimeInstalled())
                        return Fail(out status, "SteamVR/OpenVR is not installed.");

                    if (!OpenVR.IsHmdPresent())
                        return Fail(out status, "SteamVR cannot see a connected headset.");

                    var error = EVRInitError.None;
                    OpenVR.Init(ref error, EVRApplicationType.VRApplication_Background);
                    if (error != EVRInitError.None)
                        return Fail(out status, $"OpenVR initialization failed: {error}.");

                    _overlayContextHeld = true;
                }
                catch (Exception ex)
                {
                    logger.LogWarning(ex, "Could not initialize the OpenVR runtime");
                    return Fail(out status, $"OpenVR is unavailable: {ex.Message}");
                }
            }

            try
            {
                var overlay = OpenVR.Overlay;
                if (overlay == null)
                    return Fail(out status, "SteamVR initialized without an overlay interface.");

                status = _runtimeStatus = "SteamVR headset overlay ready.";
                return overlay;
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Could not acquire the OpenVR overlay interface");
                return Fail(out status, $"OpenVR overlay unavailable: {ex.Message}");
            }
        }
    }

    private CVROverlay? Fail(out string status, string message)
    {
        status = _runtimeStatus = message;
        return null;
    }

    // Registers the .vrmanifest so SteamVR knows about the app. Safe to call repeatedly;
    // returns false (never throws) when SteamVR isn't running.
    public bool AutoStart() => WithSession(RegisterManifest);

    // Ensures the manifest is registered, swallowing the unsupported-OS case.
    public void CheckIfReadyIfIsnt()
    {
        try
        {
            AutoStart();
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "DLL not found! Your current OS might not be supported for SteamVR AutoStart");
        }
    }

    //bool for checking, getting, and setting the application key for auto launch
    public bool SteamvrAutoStart
    {
        get => WithSession(apps => apps.GetApplicationAutoLaunch(ApplicationKey));
        set
        {
            var enabled = value;
            WithSession(apps =>
            {
                if (!RegisterManifest(apps))
                    return false;

                var result = apps.SetApplicationAutoLaunch(ApplicationKey, enabled);
                if (result != EVRApplicationError.None)
                {
                    logger.LogError("Failed to set SteamVR AutoStart: {0}", result);
                    return false;
                }

                return true;
            });
        }
    }

    // Opens a short-lived background OpenVR session, runs the action, then always shuts down.
    // Re-initialising per call means SteamVR being closed between calls surfaces as an init
    // error rather than crashing or hanging the process on a now-invalid native interface
    // (OpenVR.Shutdown invalidates all cached interface pointers, so a stale session that
    // outlived SteamVR must never be reused).
    private bool WithSession(Func<CVRApplications, bool> action)
    {
        lock (_sessionLock)
        {
            // An overlay is live: borrow its context and, crucially, do not shut it down. Shutting
            // down here would invalidate the overlay handle mid-calibration.
            var borrowed = _overlayContextHeld;

            if (!borrowed)
            {
                EVRInitError error = EVRInitError.None;
                OpenVR.Init(ref error, EVRApplicationType.VRApplication_Background);
                if (error != EVRInitError.None)
                {
                    logger.LogWarning("Unable to toggle autostart; SteamVR issue (Is it even running?): {0}", error);
                    return false;
                }
            }

            try
            {
                var applications = OpenVR.Applications;
                if (applications == null)
                {
                    logger.LogWarning("SteamVR applications interface is unavailable.");
                    return false;
                }

                return action(applications);
            }
            finally
            {
                if (!borrowed)
                    OpenVR.Shutdown();
            }
        }
    }

    private bool RegisterManifest(CVRApplications applications)
    {
        // Locate the manifest.vrmanifest next to the executable.
        string? fullManifestPath = Path.GetDirectoryName(AppContext.BaseDirectory);
        if (fullManifestPath == null)
        {
            logger.LogWarning("Cannot find the executable path to locate manifest.vrmanifest");
            return false;
        }

        var vrManifestPath = Path.GetFullPath(Path.Combine(fullManifestPath, "manifest.vrmanifest"));
        var registerResult = applications.AddApplicationManifest(vrManifestPath, false);
        if (registerResult != EVRApplicationError.None)
        {
            logger.LogWarning("Failed to register vrmanifest: {0}", registerResult);
            return false;
        }

        logger.LogDebug("Application installed check: {0}", applications.IsApplicationInstalled(ApplicationKey));
        return true;
    }
}
