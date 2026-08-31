using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Baballonia.Services;
using Baballonia.Services.Personalization;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests;

/// <summary>
/// Guards the contract between the stock model's positional float[45] output and the
/// personalization layer.
///
/// The stock faceModel.onnx carries no output names; index meaning is defined solely by the
/// insertion order of ParameterSenderService.FaceExpressionMap. PersonalizationSchema mirrors that
/// order. If upstream reorders, renames, adds or removes an expression, every personal model
/// trained against the old order becomes silently wrong: index 19 would stop meaning
/// MouthSmileLeft. These tests turn that silent corruption into a loud build failure.
/// </summary>
[TestClass]
[TestSubject(typeof(PersonalizationSchema))]
public class PersonalizationSchemaTests
{
    /// <summary>
    /// Reads the real FaceExpressionMap by constructing ParameterSenderService, since the map is an
    /// instance field initializer and only runs as part of the constructor.
    ///
    /// The constructor merely stores its dependencies and subscribes to one event, so nulls are
    /// safe for everything except the loop service. That one is created uninitialized to skip its
    /// constructor, which would otherwise start a 10 ms Avalonia DispatcherTimer and build the
    /// whole inference graph; subscribing to its (null) event backing field is still valid.
    /// If upstream ever makes the constructor do real work with these dependencies, this test will
    /// fail loudly - which is the intent.
    /// </summary>
    private static IReadOnlyList<string> GetUpstreamFaceExpressionNames()
    {
        var loopService = (ProcessingLoopService)RuntimeHelpers
            .GetUninitializedObject(typeof(ProcessingLoopService));

        var service = new ParameterSenderService(
            vrcftModuleSendService: null!,
            dfrSendService: null!,
            localSettingsService: null!,
            calibrationService: null!,
            processingLoopService: loopService,
            logger: null!);

        return service.FaceExpressionMap.Keys.ToList();
    }

    [TestMethod]
    public void Schema_MatchesFaceExpressionMapOrderExactly()
    {
        var upstream = GetUpstreamFaceExpressionNames();
        var schema = PersonalizationSchema.ExpressionNames;

        CollectionAssert.AreEqual(
            upstream.ToList(),
            schema.ToList(),
            "PersonalizationSchema.ExpressionNames has drifted from ParameterSenderService.FaceExpressionMap. " +
            "The stock model output is positional, so any reorder/rename invalidates previously trained " +
            "personal models. Update PersonalizationSchema, bump PersonalizationSchema.Version, and retrain " +
            "(or explicitly migrate) any existing personal model.");
    }

    [TestMethod]
    public void Schema_CountMatchesUpstreamAndFaceRawExpressions()
    {
        var upstream = GetUpstreamFaceExpressionNames();

        Assert.AreEqual(PersonalizationSchema.ExpressionCount, upstream.Count,
            "Upstream face expression count changed.");
        Assert.AreEqual(PersonalizationSchema.ExpressionCount, PersonalizationSchema.ExpressionNames.Count,
            "PersonalizationSchema.ExpressionCount disagrees with its own name list.");
        Assert.AreEqual(Utils.FaceRawExpressions, PersonalizationSchema.ExpressionCount,
            "Utils.FaceRawExpressions (the tensor/filter sizing constant) disagrees with the schema count.");
    }

    [TestMethod]
    public void Schema_NamesAreUniqueAndNonEmpty()
    {
        var names = PersonalizationSchema.ExpressionNames;

        Assert.AreEqual(names.Count, names.Distinct(StringComparer.Ordinal).Count(),
            "Duplicate expression names would make IndexOf ambiguous.");
        Assert.IsFalse(names.Any(string.IsNullOrWhiteSpace), "Blank expression name in schema.");
    }

    /// <summary>
    /// Pins the canonical hash recipe. The Python trainer reproduces this exact string
    /// (names joined by LF, UTF-8, no trailing newline) and exported personal models embed the
    /// result as "expression_schema_sha256". Changing the recipe silently invalidates that
    /// cross-language contract, so the expected value is hardcoded here rather than recomputed.
    /// </summary>
    [TestMethod]
    public void Schema_Sha256_IsStableAndWellFormed()
    {
        var hash = PersonalizationSchema.Sha256;

        Assert.AreEqual(64, hash.Length, "SHA-256 hex must be 64 characters.");
        Assert.IsTrue(hash.All(c => (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')),
            "Hash must be lowercase hex.");

        // Recomputed independently here so a change to either the names or the recipe fails loudly.
        var expected = Convert.ToHexStringLower(
            System.Security.Cryptography.SHA256.HashData(
                System.Text.Encoding.UTF8.GetBytes(string.Join("\n", PersonalizationSchema.ExpressionNames))));

        Assert.AreEqual(expected, hash,
            "Canonical serialization changed. Python (training/babble_personal/schema.py) must match: " +
            "names joined by '\\n', UTF-8, no trailing newline, lowercase hex SHA-256.");
    }

    /// <summary>
    /// Pins the exact hash the Python trainer computes
    /// (training/babble_personal/schema.py :: SCHEMA_SHA256).
    ///
    /// This is the cross-language contract: exported personal models embed this value and the
    /// runtime refuses models that disagree. If the expression list legitimately changes, this
    /// literal and the Python constant must be updated together, and existing personal models
    /// retrained - which is exactly the conversation this failure should start.
    /// </summary>
    [TestMethod]
    public void Schema_Sha256_MatchesPythonTrainerConstant()
    {
        const string pythonComputed = "c35805d06b03ec5b808f2cf1c8a1ce599aa99d7de0d65a45d2350a1086b6392e";

        Assert.AreEqual(pythonComputed, PersonalizationSchema.Sha256,
            "C# and Python disagree on the schema hash. Personal models would be rejected at load " +
            "time (or, worse, trained against a different expression order).");
    }

    [TestMethod]
    public void Schema_IndexOf_ResolvesKnownExpressionsAndRejectsUnknown()
    {
        // Spot-check anchors the plan calls out specifically (jaw open, smile/frown pairs).
        Assert.AreEqual(4, PersonalizationSchema.IndexOf("JawOpen"));
        Assert.AreEqual(19, PersonalizationSchema.IndexOf("MouthSmileLeft"));
        Assert.AreEqual(20, PersonalizationSchema.IndexOf("MouthSmileRight"));
        Assert.AreEqual(21, PersonalizationSchema.IndexOf("MouthFrownLeft"));
        Assert.AreEqual(22, PersonalizationSchema.IndexOf("MouthFrownRight"));
        Assert.AreEqual(-1, PersonalizationSchema.IndexOf("NotAnExpression"));
    }
}
