using Baballonia.Contracts;
using Microsoft.Extensions.Logging;
using System.IO;

namespace Baballonia.Services;

public class InferenceFactory
{
    private readonly ILogger<InferenceFactory> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILocalSettingsService _localSettings;

    public InferenceFactory(ILogger<InferenceFactory> logger, ILocalSettingsService localSettings,
        ILoggerFactory loggerFactory)
    {
        _logger = logger;
        _localSettings = localSettings;
        _loggerFactory = loggerFactory;
    }

    /// <param name="secondaryOutputName">
    /// Optional extra graph output to capture on every run - used for the derived face model that
    /// exposes the stock network's visual embedding. Null keeps the ordinary single-output path.
    /// </param>
    public DefaultInferenceRunner Create(string modelPath, string? secondaryOutputName = null)
    {
        var useGpu = _localSettings.ReadSetting<bool>("AppSettings_UseGPU", false);
        _loggerFactory.CreateLogger<DefaultInferenceRunner>();

        var inference = new DefaultInferenceRunner(_loggerFactory)
        {
            SecondaryOutputName = secondaryOutputName
        };

        inference.Setup(modelPath, useGpu);
        _logger.LogDebug("Loaded model {filename} with hash {ModelHash}", Path.GetFileName(modelPath), Utils.GenerateMD5(modelPath));

        return inference;
    }
}
