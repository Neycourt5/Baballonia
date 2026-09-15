using Valve.VR;

namespace Baballonia.Desktop.Calibration;

// Small native boundary so lifetime, update ordering and failure handling can be tested
// without starting SteamVR or taking ownership of a user's headset overlay.
internal interface IOpenVrOverlay
{
    EVROverlayError FindOverlay(string key, ref ulong handle);
    EVROverlayError CreateOverlay(string key, string name, ref ulong handle);
    EVROverlayError DestroyOverlay(ulong handle);
    EVROverlayError SetOverlayInputMethod(ulong handle, VROverlayInputMethod method);
    EVROverlayError SetOverlayMouseScale(ulong handle, ref HmdVector2_t scale);
    EVROverlayError SetOverlayFlag(ulong handle, VROverlayFlags flag, bool enabled);
    EVROverlayError ShowOverlay(ulong handle);
    EVROverlayError HideOverlay(ulong handle);
    bool PollNextOverlayEvent(ulong handle, ref VREvent_t vrEvent, uint size);
    EVROverlayError SetOverlayTextureBounds(ulong handle, ref VRTextureBounds_t bounds);
    EVROverlayError SetOverlayWidthInMeters(ulong handle, float width);
    EVROverlayError SetOverlayAlpha(ulong handle, float alpha);
    EVROverlayError SetOverlayTransformTrackedDeviceRelative(ulong handle, uint device, ref HmdMatrix34_t transform);
    EVROverlayError SetOverlayTexture(ulong handle, ref Texture_t texture);
    EVROverlayError ClearOverlayTexture(ulong handle);
}

internal sealed class OpenVrOverlay(CVROverlay native) : IOpenVrOverlay
{
    public EVROverlayError FindOverlay(string key, ref ulong handle) => native.FindOverlay(key, ref handle);
    public EVROverlayError CreateOverlay(string key, string name, ref ulong handle) => native.CreateOverlay(key, name, ref handle);
    public EVROverlayError DestroyOverlay(ulong handle) => native.DestroyOverlay(handle);
    public EVROverlayError SetOverlayInputMethod(ulong handle, VROverlayInputMethod method) => native.SetOverlayInputMethod(handle, method);
    public EVROverlayError SetOverlayMouseScale(ulong handle, ref HmdVector2_t scale) => native.SetOverlayMouseScale(handle, ref scale);
    public EVROverlayError SetOverlayFlag(ulong handle, VROverlayFlags flag, bool enabled) => native.SetOverlayFlag(handle, flag, enabled);
    public EVROverlayError ShowOverlay(ulong handle) => native.ShowOverlay(handle);
    public EVROverlayError HideOverlay(ulong handle) => native.HideOverlay(handle);
    public bool PollNextOverlayEvent(ulong handle, ref VREvent_t vrEvent, uint size) => native.PollNextOverlayEvent(handle, ref vrEvent, size);
    public EVROverlayError SetOverlayTextureBounds(ulong handle, ref VRTextureBounds_t bounds) => native.SetOverlayTextureBounds(handle, ref bounds);
    public EVROverlayError SetOverlayWidthInMeters(ulong handle, float width) => native.SetOverlayWidthInMeters(handle, width);
    public EVROverlayError SetOverlayAlpha(ulong handle, float alpha) => native.SetOverlayAlpha(handle, alpha);
    public EVROverlayError SetOverlayTransformTrackedDeviceRelative(ulong handle, uint device, ref HmdMatrix34_t transform) => native.SetOverlayTransformTrackedDeviceRelative(handle, device, ref transform);
    public EVROverlayError SetOverlayTexture(ulong handle, ref Texture_t texture) => native.SetOverlayTexture(handle, ref texture);
    public EVROverlayError ClearOverlayTexture(ulong handle) => native.ClearOverlayTexture(handle);
}
