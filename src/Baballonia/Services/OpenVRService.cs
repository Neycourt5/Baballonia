using Microsoft.Extensions.Logging;
using System;
using System.IO;
using Valve.VR;

namespace Baballonia.Services;

public class OpenVRService(ILogger<OpenVRService> logger)
{
    //app key needed for vrmanifest
    private const string ApplicationKey = "projectbabble.Baballonia";
    private readonly object _initializationLock = new();
    private bool _isAutoStartReady;
    private bool _isInitialized;
    private string _runtimeStatus = "SteamVR has not been checked yet.";

    public string RuntimeStatus => _runtimeStatus;

    /// <summary>
    /// Initializes the one process-wide OpenVR background context and returns its overlay API.
    /// Calibration presenters use this instead of launching or mirroring another desktop window.
    /// </summary>
    public CVROverlay? TryGetOverlay(out string status)
    {
        lock (_initializationLock)
        {
            if (!_isInitialized)
            {
                try
                {
                    if (!OpenVR.IsRuntimeInstalled())
                    {
                        status = _runtimeStatus = "SteamVR/OpenVR is not installed.";
                        return null;
                    }

                    if (!OpenVR.IsHmdPresent())
                    {
                        status = _runtimeStatus = "SteamVR cannot see a connected headset.";
                        return null;
                    }

                    var error = EVRInitError.None;
                    OpenVR.Init(ref error, EVRApplicationType.VRApplication_Background);
                    if (error != EVRInitError.None)
                    {
                        status = _runtimeStatus = $"OpenVR initialization failed: {error}.";
                        return null;
                    }

                    _isInitialized = true;
                }
                catch (Exception ex)
                {
                    logger.LogWarning(ex, "Could not initialize the OpenVR runtime");
                    status = _runtimeStatus = $"OpenVR is unavailable: {ex.Message}";
                    return null;
                }
            }

            try
            {
                var overlay = OpenVR.Overlay;
                if (overlay == null)
                {
                    status = _runtimeStatus = "SteamVR initialized without an overlay interface.";
                    return null;
                }

                status = _runtimeStatus = "SteamVR headset overlay ready.";
                return overlay;
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Could not acquire the OpenVR overlay interface");
                status = _runtimeStatus = $"OpenVR overlay unavailable: {ex.Message}";
                return null;
            }
        }
    }

    //AutoStart function
    public bool AutoStart()
    {
        // The process owns one OpenVR context. Reusing it also avoids disrupting an active
        // calibration overlay when the Settings page checks autostart.
        if (TryGetOverlay(out var runtimeStatus) == null)
        {
            logger.LogWarning("Failed to Enable SteamVR AutoStart: {Status}", runtimeStatus);
            _isAutoStartReady = false;
            return _isAutoStartReady;
        }

        // Trying to check for and find the manifest.vrmanifest file using the exe's directory
        string? fullManifestPath = Path.GetDirectoryName(AppContext.BaseDirectory);
        if (fullManifestPath == null)
        {
            throw new Exception("Can not find the executable Path");
        }
        var VRManifestPath = Path.GetFullPath(Path.Combine(fullManifestPath, "manifest.vrmanifest"));

        // Checking if the manifest is registered and if anything went wrong
        var VRManifestRegResult = OpenVR.Applications.AddApplicationManifest(VRManifestPath, false);
        if(VRManifestRegResult != EVRApplicationError.None)
        {
            logger.LogWarning("Failed to register vrmanifest: {0}", VRManifestRegResult);
            _isAutoStartReady = false;
            return _isAutoStartReady;
        }
        // Checking if the application in the vrmanifest is valid
        var ApplicationCheck = OpenVR.Applications.IsApplicationInstalled(ApplicationKey);
        logger.LogDebug("checking for application {0}", ApplicationCheck);

        logger.LogInformation("Successfully Added to SteamVR startup apps");

        _isAutoStartReady = true;
        return _isAutoStartReady;
    }

    // Checking to see if autostart is ready
    public void CheckIfReadyIfIsnt()
    {
        if (_isAutoStartReady) return;

        try
        {
            AutoStart();
        }
        catch (Exception e)
        {
            logger.LogWarning("DLL not found! Your current OS might not be supported for SteamVR AutoStart", e);
        }
    }

    //bool for checking, getting, and setting the application key for auto launch
    public bool SteamvrAutoStart
    {
        get => _isAutoStartReady && OpenVR.Applications.GetApplicationAutoLaunch(ApplicationKey);
        set
        {
            if (!_isAutoStartReady && !AutoStart())
            {
                logger.LogError("Failed to change SteamVR AutoStart setting. OpenVR could not be Configured.");
                return;
            }

            var SetAutoStartResult = OpenVR.Applications.SetApplicationAutoLaunch(ApplicationKey, value);
            if (SetAutoStartResult != EVRApplicationError.None)
            {
                logger.LogError("Failed to set SteamVR Auto Start: {0}", SetAutoStartResult);
            }
        }
    }
}
