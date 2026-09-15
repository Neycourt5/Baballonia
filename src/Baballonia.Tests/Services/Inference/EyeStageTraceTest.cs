using System.Linq;
using Baballonia.Services.Inference;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.Services.Inference;

[TestClass]
public class EyeStageTraceTest
{
    [TestMethod]
    public void DisabledTraceIsEmptyAndEnabledTraceIsBoundedOwnedAndOrdered()
    {
        var trace=new EyeStageTrace(16);
        var map=new OrderedFloatMap(["/leftEyeX","/rightEyeX"]);
        trace.Add(1,"raw",map); Assert.AreEqual(0,trace.Count);
        trace.Start();
        for(var i=0;i<30;i++) { map["/leftEyeX"]=i; trace.Add(i,"raw",map); }
        map["/leftEyeX"]=999;
        var rows=trace.Snapshot(); Assert.AreEqual(16,rows.Length);
        Assert.AreEqual(14,rows.First().Frame); Assert.AreEqual(29f,rows.Last().Values["/leftEyeX"]);
        trace.Stop(); trace.Add(40,"raw",map); Assert.AreEqual(29,trace.Snapshot().Last().Frame);
    }

    [TestMethod]
    public void NonfiniteValuesAndUnknownSourceRemainExplicitlyMissing()
    {
        var trace=new EyeStageTrace(); trace.Start();
        var map=new OrderedFloatMap(["/leftEyeX"]); map["/leftEyeX"]=float.NaN;
        trace.Add(1,"raw",map);
        Assert.IsNull(trace.Snapshot()[0].Values["/leftEyeX"]);
        Assert.IsNull(trace.Snapshot()[0].Source);
    }
}
