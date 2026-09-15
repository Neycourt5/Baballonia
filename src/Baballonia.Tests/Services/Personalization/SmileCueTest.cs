using System.Linq;
using Baballonia.Services.Personalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Personalization;

/// <summary>
/// The smile cues, and the gap in the corpus they exist to close.
/// </summary>
/// <remarks>
/// An ordinary smile was arriving on the avatar as a wide open-mouthed grin. Reading what the cues
/// actually teach explains it: <see cref="GuidedCues.Smile"/> drives only the two corners, so JawOpen
/// is uncued and correctly carries no weight, and <see cref="GuidedCues.SmileWithJaw"/> drives the
/// jaw at 0.6. Across the whole corpus the only jaw supervision that ever co-occurred with a smile
/// said the jaw was **open** - the model was never once shown a smile with the teeth together.
///
/// These tests pin the fix, and they check supervision rather than appearance: a dimension commanded
/// to zero is only useful if it is genuinely cued, because an uncued zero is indistinguishable from
/// no opinion at all.
/// </remarks>
[TestClass]
public class SmileCueTest
{
    private static int Ix(string name) => PersonalizationSchema.IndexOf(name);

    [TestMethod]
    public void TheJawIsCuedRatherThanMerelyLeftAtZero()
    {
        // The whole point. target[JawOpen] == 0 means nothing unless JawOpen is in Dims - the
        // labeller weights cued dimensions, and an uncued one is "no opinion", not "shut".
        foreach (var cue in new[] { GuidedCues.SmileJawShut, GuidedCues.SmileTeeth })
        {
            CollectionAssert.Contains(cue.Dims.ToArray(), Ix("JawOpen"),
                $"{cue.Id} must cue the jaw for the zero to carry any weight");

            foreach (var level in cue.EffectiveLevels)
            {
                Assert.AreEqual(0f, cue.TargetAt(level)[Ix("JawOpen")],
                    $"{cue.Id} at {level} should command the jaw shut");
            }
        }
    }

    [TestMethod]
    public void TheSmileItselfStillRisesWithTheLevel()
    {
        // Pinning the jaw must not flatten the expression being taught: the corners still have to
        // scale, or the cue teaches one size of smile and the original complaint was about size.
        var left = Ix("MouthSmileLeft");

        foreach (var cue in new[] { GuidedCues.SmileJawShut, GuidedCues.SmileTeeth })
        {
            var levels = cue.EffectiveLevels;
            Assert.IsTrue(levels.Count >= 2, $"{cue.Id} should teach more than one size");

            var half = cue.TargetAt(levels[0])[left];
            var full = cue.TargetAt(levels[^1])[left];

            Assert.IsTrue(full > half,
                $"{cue.Id} commands {half} then {full} - the corners are not scaling");
        }
    }

    [TestMethod]
    public void ShowingTeethLiftsTheLipRatherThanOpeningTheMouth()
    {
        // Teeth appear because the upper lip rises, not because the jaw drops. Cueing the jaw here
        // instead would teach exactly the fault being fixed.
        var cue = GuidedCues.SmileTeeth;
        var target = cue.TargetAt(1.0f);

        Assert.IsTrue(target[Ix("MouthUpperUpLeft")] > 0f, "the upper lip should lift");
        Assert.IsTrue(target[Ix("MouthUpperUpRight")] > 0f, "the upper lip should lift");
        Assert.AreEqual(0f, target[Ix("JawOpen")], "and the jaw should stay shut");

        Assert.IsTrue(target[Ix("MouthUpperUpLeft")] < target[Ix("MouthSmileLeft")],
            "a smile that bares the full gum is a grimace, not an everyday smile");
    }

    [TestMethod]
    public void TheCorpusStillContainsASmileThatOpensTheJaw()
    {
        // The other half. Teaching only jaw-shut smiles would trade one wrong face for another: a
        // real laugh would then read as a closed mouth.
        var open = GuidedCues.SmileWithJaw.TargetAt(1.0f)[Ix("JawOpen")];

        Assert.IsTrue(open > 0f,
            "something must still teach that a smile can open the mouth");
    }

    [TestMethod]
    public void TheSmilePassEndsOnTheOpenJaw()
    {
        // Ordering is deliberate: finish having taught that a smile *can* open the mouth, rather
        // than that it never does.
        var pass = GuidedCues.SmilePass;

        Assert.AreEqual(GuidedCues.SmileWithJaw.Id, pass[^1].Id,
            "the open-jaw smile should come last");

        Assert.IsTrue(
            pass.Take(pass.Count - 1).All(cue => cue.TargetAt(1.0f)[Ix("JawOpen")] == 0f),
            "everything before it should hold the jaw shut");
    }

    [TestMethod]
    public void TheNewCuesAreOfferedAndFindableById()
    {
        // A cue missing from All is invisible in the picker and cannot be restored from a saved
        // selection, which is a silent way for a routine to stop existing.
        foreach (var cue in new[] { GuidedCues.SmileJawShut, GuidedCues.SmileTeeth })
        {
            CollectionAssert.Contains(GuidedCues.All.ToArray(), cue, $"{cue.Id} is not offered");
            Assert.AreEqual(cue, GuidedCues.ById(cue.Id), $"{cue.Id} cannot be restored by id");
        }
    }

    [TestMethod]
    public void ARoutineExistsForFixingSmilesOnTheirOwn()
    {
        var routine = GuidedRoutineChoice.All.SingleOrDefault(choice => choice.Id == "smiles");

        Assert.IsNotNull(routine, "there should be a smile-only routine in the picker");
        CollectionAssert.AreEqual(GuidedCues.SmilePass.ToArray(), routine!.Cues.ToArray());
    }

    [TestMethod]
    public void EveryCueDrivesRealExpressions()
    {
        // A misspelled name resolves to -1, which TargetAt skips silently - the cue would run,
        // record, and teach nothing, with no error anywhere.
        foreach (var cue in GuidedCues.All)
        {
            Assert.IsFalse(cue.Dims.Any(dim => dim < 0),
                $"{cue.Id} names an expression that does not exist");
        }
    }
}

/// <summary>
/// The smile candidate lab: the same preview machinery as grimace, asking a different question.
/// </summary>
/// <remarks>
/// Grimace asks what an expression *is*. This asks how much of one there should be, because the
/// complaint was an ordinary smile arriving as a huge grin. The candidates therefore vary intensity
/// rather than identity, and the checks here are mostly that they cannot quietly become a grimace.
/// </remarks>
[TestClass]
public class SmileCandidateCatalogTest
{
    private static int Ix(string name) => PersonalizationSchema.IndexOf(name);

    [TestMethod]
    public void EveryCandidateIsASmileOfADifferentSize()
    {
        var candidates = SmileCandidateCatalog.All;
        Assert.IsTrue(candidates.Count >= 3, "a size question needs more than two answers");

        var corners = candidates
            .Select(candidate => candidate.CreateTarget()[Ix("MouthSmileLeft")])
            .ToList();

        Assert.IsTrue(corners.All(value => value > 0f), "every candidate must actually smile");
        Assert.AreEqual(corners.Count, corners.Distinct().Count(),
            "two candidates with the same smile intensity offer no choice");
    }

    [TestMethod]
    public void TeethComeFromTheLipNotTheJaw()
    {
        // The fault being fixed is the mouth hanging open. A candidate set that opened the jaw to
        // show teeth would be offering the user four flavours of the same complaint.
        foreach (var candidate in SmileCandidateCatalog.All)
        {
            var target = candidate.CreateTarget();

            Assert.IsTrue(target[Ix("MouthUpperUpLeft")] > 0f,
                $"{candidate.Id} shows no teeth");
            Assert.IsTrue(target[Ix("JawOpen")] < 0.2f,
                $"{candidate.Id} opens the jaw to {target[Ix("JawOpen")]}, which is the original complaint");
        }
    }

    [TestMethod]
    public void MostCandidatesKeepTheJawCompletelyShut()
    {
        var shut = SmileCandidateCatalog.All
            .Count(candidate => candidate.CreateTarget()[Ix("JawOpen")] == 0f);

        Assert.IsTrue(shut >= SmileCandidateCatalog.All.Count - 1,
            "at most one candidate should part the teeth at all");
    }

    [TestMethod]
    public void JawShutCandidatesCarryExplicitZeroSupervisionIntoTheConfirmedCue()
    {
        var jaw = Ix("JawOpen");
        foreach (var candidate in SmileCandidateCatalog.All
                     .Where(item => item.CreateTarget()[jaw] == 0f))
        {
            CollectionAssert.Contains(candidate.Dims.ToArray(), jaw,
                $"{candidate.Id} leaves JawOpen uncued instead of commanding it shut");
            CollectionAssert.Contains(candidate.SuppressedDims.ToArray(), jaw,
                $"{candidate.Id} does not identify JawOpen as a deliberate zero");

            var cue = candidate.CreateConfirmedCue();
            CollectionAssert.Contains(cue.Dims.ToArray(), jaw,
                $"{candidate.Id} loses JawOpen when it becomes training supervision");
            CollectionAssert.Contains(cue.SuppressedDims.ToArray(), jaw,
                $"{candidate.Id} loses the deliberate-zero marker when confirmed");
            Assert.AreEqual(0f, cue.TargetAt(1f)[jaw],
                $"{candidate.Id} no longer commands a shut jaw after confirmation");
        }
    }

    [TestMethod]
    public void ConfirmingOneProducesASmileCueRatherThanAGrimaceOne()
    {
        // Silent and expensive if wrong: the user performs whatever the instruction says, and it is
        // recorded under the label of the pose they confirmed.
        foreach (var candidate in SmileCandidateCatalog.All)
        {
            var action = candidate.CreateConfirmedCue().Action.ToLowerInvariant();

            StringAssert.Contains(action, "smile", $"{candidate.Id} does not ask for a smile");
            Assert.IsFalse(action.Contains("grimace"),
                $"{candidate.Id} instructs a grimace while claiming to be a smile");
        }
    }

    [TestMethod]
    public void TheTwoLabsRememberTheirChoicesSeparately()
    {
        // One settings key for both would mean confirming a smile silently discarded the confirmed
        // grimace, and the window would go on showing it as confirmed.
        Assert.AreNotEqual(
            GrimaceCandidateCatalog.Catalog.ConfirmationSetting,
            SmileCandidateCatalog.Catalog.ConfirmationSetting,
            "the labs must not share a settings key");

        Assert.AreNotEqual(
            GrimaceCandidateCatalog.Catalog.ConfirmedRoutineId,
            SmileCandidateCatalog.Catalog.ConfirmedRoutineId,
            "the labs must not replace each other's guided routine");
        Assert.AreEqual("smile-confirmed", SmileCandidateCatalog.Catalog.ConfirmedRoutineId);
    }

    [TestMethod]
    public void CandidateIdsAreUniqueAcrossBothLabs()
    {
        // Ids reach the recorder and the run manifests, where a collision would make two different
        // poses indistinguishable after the fact.
        var all = GrimaceCandidateCatalog.All.Concat(SmileCandidateCatalog.All)
            .Select(candidate => candidate.Id)
            .ToList();

        Assert.AreEqual(all.Count, all.Distinct().Count(), "candidate ids collide between labs");
    }

    [TestMethod]
    public void EveryCandidateOnlyDrivesRealExpressions()
    {
        foreach (var candidate in SmileCandidateCatalog.All)
        {
            Assert.IsFalse(candidate.Dims.Any(dim => dim < 0),
                $"{candidate.Id} names an expression that does not exist");
        }
    }
}
