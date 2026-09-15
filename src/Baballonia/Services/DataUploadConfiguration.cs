using System;

namespace Baballonia.Services;

/// <summary>
/// Optional upload credentials supplied by the person operating the app.
/// No credentials are bundled, persisted in a profile, or included in diagnostics.
/// </summary>
public sealed class DataUploadConfiguration
{
    public const string AccessKeyEnvironmentVariable = "BABALLONIA_UPLOAD_ACCESS_KEY_ID";
    public const string SecretKeyEnvironmentVariable = "BABALLONIA_UPLOAD_SECRET_ACCESS_KEY";

    internal string? AccessKeyId { get; }
    internal string? SecretAccessKey { get; }
    public bool IsConfigured => !string.IsNullOrWhiteSpace(AccessKeyId) &&
                                !string.IsNullOrWhiteSpace(SecretAccessKey);

    private DataUploadConfiguration(string? accessKeyId, string? secretAccessKey)
    {
        AccessKeyId = accessKeyId;
        SecretAccessKey = secretAccessKey;
    }

    public static DataUploadConfiguration FromEnvironment(Func<string, string?>? read = null)
    {
        read ??= Environment.GetEnvironmentVariable;
        return new DataUploadConfiguration(read(AccessKeyEnvironmentVariable), read(SecretKeyEnvironmentVariable));
    }
}
