using System;
using System.Reflection;
using Baballonia.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Baballonia.Tests.ViewModels;

[TestClass]
public class MainViewModelActivationTest
{
    [TestMethod]
    public void PageActivationHonorsKeyedConstructorDependencies()
    {
        var unkeyed = new Marker("unkeyed");
        var keyed = new Marker("smile");
        var services = new ServiceCollection();
        services.AddSingleton(unkeyed);
        services.AddKeyedSingleton("smile", keyed);
        using var provider = services.BuildServiceProvider();

        var activate = typeof(MainViewModel).GetMethod(
            "CreateInstance",
            BindingFlags.Static | BindingFlags.NonPublic);
        Assert.IsNotNull(activate, "the navigation activation boundary could not be inspected");

        var probe = (ActivationProbe)activate.Invoke(
            null,
            [provider, typeof(ActivationProbe)])!;

        Assert.AreSame(unkeyed, probe.Unkeyed);
        Assert.AreSame(keyed, probe.Keyed,
            "keyed constructor metadata was ignored and the unkeyed service was injected twice");
    }

    /// <summary>
    /// The same activation, through the provider production actually hands it.
    /// </summary>
    /// <remarks>
    /// The test above passes a <see cref="ServiceProvider"/> directly, which implements
    /// <c>IKeyedServiceProvider</c> and therefore satisfies <c>[FromKeyedServices]</c>. Navigation
    /// passed <c>Ioc.Default</c>, which does not - so the page that used keyed metadata threw on
    /// open while every test went on passing.
    /// </remarks>
    [TestMethod]
    public void TheProviderNavigationActuallyUsesCanResolveKeyedDependencies()
    {
        var services = new ServiceCollection();
        services.AddSingleton(new Marker("unkeyed"));
        services.AddKeyedSingleton("smile", new Marker("smile"));
        using var provider = services.BuildServiceProvider();

        var navigationProvider = MainViewModel.ActivationServices(provider);

        Assert.IsInstanceOfType<IKeyedServiceProvider>(navigationProvider,
            "navigation must activate pages through a provider that understands keyed services");

        var activate = typeof(MainViewModel).GetMethod(
            "CreateInstance", BindingFlags.Static | BindingFlags.NonPublic)!;

        var probe = (ActivationProbe)activate.Invoke(
            null, [navigationProvider, typeof(ActivationProbe)])!;

        Assert.AreEqual("smile", probe.Keyed.Name);
    }

    /// <summary>
    /// A provider shaped like <c>Ioc.Default</c>: delegates everything, implements only
    /// <see cref="IServiceProvider"/>.
    /// </summary>
    /// <remarks>
    /// Reproduces the production failure without touching the process-wide Ioc singleton, which
    /// can only be configured once and would make this test order-dependent.
    /// </remarks>
    private sealed class UnkeyedWrapper(IServiceProvider inner) : IServiceProvider
    {
        public object? GetService(Type serviceType) => inner.GetService(serviceType);
    }

    [TestMethod]
    public void AProviderThatCannotDoKeyedLookupsIsExchangedForOneThatCan()
    {
        var services = new ServiceCollection();
        services.AddSingleton(new Marker("unkeyed"));
        services.AddKeyedSingleton("smile", new Marker("smile"));
        using var provider = services.BuildServiceProvider();

        var wrapper = new UnkeyedWrapper(provider);
        Assert.IsNotInstanceOfType<IKeyedServiceProvider>(wrapper,
            "the wrapper must reproduce the shape that broke, or this proves nothing");

        var activate = typeof(MainViewModel).GetMethod(
            "CreateInstance", BindingFlags.Static | BindingFlags.NonPublic)!;

        // Through the wrapper directly: this is what navigation used to do, and it fails.
        var direct = Assert.Throws<TargetInvocationException>(() =>
            activate.Invoke(null, [wrapper, typeof(ActivationProbe)]));
        Assert.IsInstanceOfType<InvalidOperationException>(direct.InnerException,
            "a keyed dependency should be unresolvable through an unkeyed provider");

        // Through the exchange: the page opens.
        var probe = (ActivationProbe)activate.Invoke(
            null, [MainViewModel.ActivationServices(wrapper), typeof(ActivationProbe)])!;

        Assert.AreEqual("smile", probe.Keyed.Name);
        Assert.AreEqual("unkeyed", probe.Unkeyed.Name);
    }

    public sealed record Marker(string Name);

    public sealed class ActivationProbe(
        Marker unkeyed,
        [FromKeyedServices("smile")] Marker keyed)
    {
        public Marker Unkeyed { get; } = unkeyed;
        public Marker Keyed { get; } = keyed;
    }
}
