using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Baballonia.Contracts;
using Baballonia.Services.Inference;
using Baballonia.Services.Inference.Models;

namespace Baballonia.Services.Personalization.C2;

/// <summary>The same reference/output checks apply to capture, trials and restored selections.</summary>
public sealed class C2RuntimeContext(PersonalModelManager models, ICalibrationService calibration,
    ILocalSettingsService settings, FacePipelineManager face)
{
    private sealed record CameraContext(string? Address, string? Backend, CameraSettings? Transform);

    public async Task<string> PrepareContractAsync(bool requireAudioOff = true)
    {
        if (requireAudioOff && settings.ReadSetting<bool>("AudioAssist_Enabled"))
            throw new InvalidOperationException("Turn Audio Assist off for C2 recording, training or a comparison trial. Keep using this C2 supports Audio Assist for everyday use.");
        if (!models.IsActive || models.LoadedMetadata?.AdapterType != TrainingModelChoice.SharedFeaturesAdapterType ||
            !models.EmbeddingRunnerAvailable || models.ActiveModelPath == null)
            throw new InvalidOperationException("Select your working Model C and start its feature runner first.");
        var output = C2OutputContext.Capture(calibration, settings);
        output.Validate();
        var reference = models.ActiveModelPath;
        var blend = models.Blend;
        var stock = Path.Combine(AppContext.BaseDirectory, "faceModel.onnx");
        var producer = EmbeddingModelStore.TryGetValid(stock);
        if (!producer.Valid || producer.Path == null) throw new InvalidOperationException(producer.Message);
        var camera = JsonSerializer.Serialize(new CameraContext(
            settings.ReadSetting<string>("LastOpenedFaceCamera"),
            settings.ReadSetting<string>("LastOpenedPreferredCaptureFaceCamera"),
            settings.ReadSetting<CameraSettings>("FaceCamera")));
        return await Task.Run(() =>
        {
            var contract = new C2ReferenceContract(stock, C2Contract.Hash(stock), producer.Path,
                C2Contract.Hash(producer.Path), reference, C2Contract.Hash(reference), blend, output, camera,
                SchemaSha256: PersonalizationSchema.Sha256);
            var json = JsonSerializer.SerializeToUtf8Bytes(contract, PersonalizationPaths.IndentedJson);
            var hash = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(json));
            var path = Path.Combine(C2Contract.Root, "Contracts", hash + ".json");
            if (!File.Exists(path)) C2Contract.WriteAtomic(path, contract);
            return path;
        });
    }

    public Func<bool> ContextGuard(C2ReferenceContract contract)
    {
        var failure = ContextFailure(contract);
        return () => failure() == null;
    }

    public Func<string?> ContextFailure(C2ReferenceContract contract, bool allowAudioAssist = false,
        Func<int, bool>? correctsChannel = null)
    {
        var camera = JsonSerializer.Deserialize<CameraContext>(contract.CameraSettings)
            ?? throw new InvalidDataException("C2 camera settings are missing.");
        return () =>
        {
            // Compare saved values, not the transform object captured before camera startup.
            // A kept candidate may load while Home has not yet connected the camera.
            if (!models.IsActive || models.ActiveModelPath != contract.ReferencePath)
                return "The reference Model C is no longer active. Select the Model C used to train this candidate, then choose Keep using this C2.";
            if (models.Blend != contract.ReferenceBlend)
                return "Model C blend strength changed. Restore its training strength, then choose Keep using this C2.";
            if ((face.SourceInstalledAtUtc != null && face.ActiveTransformation != camera.Transform) ||
                settings.ReadSetting<CameraSettings>("FaceCamera") != camera.Transform)
                return "Face camera crop, rotation or image settings changed. Restore the recorded settings before using this C2.";
            if (settings.ReadSetting<string>("LastOpenedFaceCamera") != camera.Address ||
                settings.ReadSetting<string>("LastOpenedPreferredCaptureFaceCamera") != camera.Backend)
                return "The face camera or capture backend changed. Restore the camera used to record this C2.";
            if (!allowAudioAssist && settings.ReadSetting<bool>("AudioAssist_Enabled"))
                return "Audio Assist is on. Turn it off for recording or a comparison trial, or choose Keep using this C2 for everyday use with audio.";
            if (settings.ReadSetting<bool>("AppSettings_OneEuroEnabled") != contract.Output.FilterEnabled ||
                settings.ReadSetting<float>("AppSettings_OneEuroMinFreqCutoff") != contract.Output.MinCutoff ||
                settings.ReadSetting<float>("AppSettings_OneEuroSpeedCutoff") != contract.Output.Beta)
                return "Face smoothing settings changed. Restore the settings used to train this C2.";
            for (var i = 0; i < 45; i++)
            {
                if (correctsChannel?.Invoke(i) == false) continue;
                var current = calibration.GetExpressionSettings(PersonalizationSchemaBinding.ExpectedKeys[i]);
                var expected = contract.Output.Ranges[i];
                if (current.Lower != expected[0] || current.Upper != expected[1] || current.Min != expected[2] || current.Max != expected[3])
                    return $"Face calibration for {PersonalizationSchema.ExpressionNames[i]} changed. Restore its recorded range before using this C2.";
            }
            return null;
        };
    }

    public async Task<C2PreparedCandidate> PrepareCandidateAsync(string directory, bool keep)
    {
        // C2 consumes the visual reference BEFORE Audio Assist. Everyday use can keep audio on;
        // collection and comparison trials still exclude it because the report has no audio replay.
        var contractPath = await PrepareContractAsync(requireAudioOff: !keep);
        var current = JsonSerializer.Deserialize<C2ReferenceContract>(await File.ReadAllTextAsync(contractPath))!;
        var contract = JsonSerializer.Deserialize<C2ReferenceContract>(await File.ReadAllTextAsync(Path.Combine(directory, "contract.json")))!;
        // Training stores a JSON copy of the original C# contract. Reconstitute its canonical
        // bytes so the report/model still bind to the exact context that produced this candidate.
        var bytes = JsonSerializer.SerializeToUtf8Bytes(contract, PersonalizationPaths.IndentedJson);
        var contextHash = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes));
        var modelPath = ValidateCandidate(directory, contextHash);
        var support = ReadSupportedChannels(directory);
        var corrector = await Task.Run(() => new C2CandidateCorrector(modelPath, contextHash, contract.EmbeddingDim, support));
        try
        {
            // Comparison and collection bind every saved output range. Everyday use only needs
            // the ranges C2 changes; all other channels are explicitly preserved from Model C.
            Func<int, bool>? correctsChannel = keep ? corrector.CorrectsChannel : null;
            ValidateContext(contract, current, correctsChannel);
            var failure = ContextFailure(contract, allowAudioAssist: keep, correctsChannel: correctsChannel);
            return new C2PreparedCandidate(corrector, () =>
            {
                var reason = failure();
                if (reason != null) corrector.Invalidate(reason);
                return reason == null;
            });
        }
        catch { corrector.Dispose(); throw; }
    }

    public static void ValidateContext(C2ReferenceContract recorded, C2ReferenceContract current,
        Func<int, bool>? correctsChannel = null)
    {
        recorded.Output.Validate();
        current.Output.Validate();
        // Updating/moving the application can relocate the identical stock/feature files. Match
        // their content hashes while keeping the user's reference C, blend and settings exact.
        var relocated = current with { StockPath = recorded.StockPath, FeaturePath = recorded.FeaturePath };
        for (var i = 0; i < 45; i++)
        {
            if (correctsChannel?.Invoke(i) == false) continue;
            if (!recorded.Output.Ranges[i].SequenceEqual(current.Output.Ranges[i]))
                throw new InvalidOperationException($"Face calibration for {PersonalizationSchema.ExpressionNames[i]} changed. Restore its recorded range before using this C2.");
        }
        if (correctsChannel != null)
            relocated = relocated with { Output = current.Output with { Ranges = recorded.Output.Ranges } };
        if (JsonSerializer.Serialize(recorded) != JsonSerializer.Serialize(relocated))
            throw new InvalidOperationException("The reference model or C2 settings changed. Restore the settings used to train this C2 or train a new candidate.");
    }

    public static IReadOnlySet<int>? ReadSupportedChannels(string directory)
    {
        using var report = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "summary.json")));
        // Reports without a support list retain the original strict compatibility policy.
        if (!report.RootElement.TryGetProperty("support", out var support)) return null;
        if (support.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException("C2 supported expressions are invalid.");
        var indices = new HashSet<int>();
        var names = PersonalizationSchema.ExpressionNames.ToArray();
        foreach (var expression in support.EnumerateArray())
        {
            var index = expression.ValueKind == JsonValueKind.String
                ? Array.IndexOf(names, expression.GetString()) : -1;
            if (index < 0 || !indices.Add(index))
                throw new InvalidDataException("C2 supported expressions are invalid.");
        }
        if (indices.Count == 0) throw new InvalidDataException("C2 supported expressions are missing.");
        return indices;
    }

    public static string ValidateCandidate(string directory, string contextHash)
    {
        using var report = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "summary.json")));
        if (report.RootElement.GetProperty("context_sha256").GetString() != contextHash)
            throw new InvalidOperationException("The reference/output settings changed. Restore the settings used to train this C2 or train a new candidate.");
        var modelPath = Path.Combine(directory, "candidate.onnx");
        if (C2Contract.Hash(modelPath) != report.RootElement.GetProperty("candidate_sha256").GetString())
            throw new InvalidDataException("Candidate changed after comparison; it cannot be activated.");
        return modelPath;
    }
}
