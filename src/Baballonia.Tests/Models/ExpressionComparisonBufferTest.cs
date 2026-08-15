using Baballonia.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Models;

[TestClass]
public sealed class ExpressionComparisonBufferTest
{
    [TestMethod]
    public void SwitchingToStock_MirrorsNewestRawFrameInsteadOfFreezingCorrection()
    {
        var buffer = new ExpressionComparisonBuffer(3);

        buffer.AcceptRaw([0.1f, 0.2f, 0.3f], personalCorrectorActive: true);
        buffer.AcceptCorrected([0.1f, 0.2f, 0.3f], [0.7f, 0.8f, 0.9f]);
        buffer.AcceptRaw([0.4f, 0.5f, 0.6f], personalCorrectorActive: false);

        Assert.AreEqual(0.4f, buffer.StockAt(0), 1e-6f);
        Assert.AreEqual(0.4f, buffer.PersonalAt(0), 1e-6f);
        Assert.AreEqual(0.6f, buffer.StockAt(2), 1e-6f);
        Assert.AreEqual(0.6f, buffer.PersonalAt(2), 1e-6f);
    }

    [TestMethod]
    public void ActiveCorrector_RawAdvancesStockAndCorrectedAdvancesPersonal()
    {
        var buffer = new ExpressionComparisonBuffer(2);
        buffer.AcceptRaw([0.2f, 0.3f], personalCorrectorActive: false);

        buffer.AcceptRaw([0.4f, 0.5f], personalCorrectorActive: true);
        Assert.AreEqual(0.4f, buffer.StockAt(0), 1e-6f);
        Assert.AreEqual(0.2f, buffer.PersonalAt(0), 1e-6f,
            "Personal remains the prior complete snapshot until its matching corrected event arrives.");

        buffer.AcceptCorrected([0.4f, 0.5f], [0.6f, 0.7f]);
        Assert.AreEqual(0.4f, buffer.StockAt(0), 1e-6f);
        Assert.AreEqual(0.6f, buffer.PersonalAt(0), 1e-6f);
    }
}
