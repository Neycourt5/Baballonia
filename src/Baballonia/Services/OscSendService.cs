using Baballonia.Contracts;
using Microsoft.Extensions.Logging;
using OscCore;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace Baballonia.Services;

/// <summary>
/// OscSendService is responsible for encoding osc messages and sending them over OSC
/// </summary>
public abstract class OscSendService(
    ILogger<OscSendService> logger,
    IOscTarget oscTarget)
{
    public event Action<int> OnMessagesDispatched = _ => { };
    protected readonly IOscTarget OscTarget = oscTarget;
    private Socket _sendSocket;
    private IPEndPoint? _destination;

    /// <summary>The UDP endpoint currently selected for this transport.</summary>
    public string Destination => _destination?.ToString() ?? "unconfigured";

    protected void UpdateTarget(IPEndPoint endpoint)
    {
        _sendSocket?.Close();
        OscTarget.IsConnected = false;
        _destination = endpoint;

        _sendSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);

        try
        {
            _sendSocket.Connect(endpoint);
            OscTarget.IsConnected = true;
        }
        catch (SocketException ex)
        {
            logger.LogWarning("Failed to bind to sender endpoint: {IpEndPoint}. {ExMessage}", endpoint, ex.Message);
        }
        catch (Exception ex)
        {
            logger.LogError("Unexpected Exception while binding to sender endpoint: {IpEndPoint}. {ExMessage}", endpoint, ex.Message);
        }
    }

    public virtual async Task<OscDispatchResult> Send(OscMessage message, CancellationToken ct)
    {
        var destination = Destination;
        if (_sendSocket is not { Connected: true } socket)
            return FailedDispatch(destination, 1, "OSC UDP socket is not connected.");

        try
        {
            await socket.SendAsync(message.ToByteArray(), SocketFlags.None, ct);
            OnMessagesDispatched(1);
            return SuccessfulDispatch(destination, 1);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error sending OSC message");
            return FailedDispatch(destination, 1, ex.Message);
        }
    }

    public virtual async Task<OscDispatchResult> Send(OscMessage[] messages, CancellationToken ct)
    {
        var destination = Destination;
        if (messages.Length == 0)
            return SuccessfulDispatch(destination, 0);

        if (_sendSocket is not { Connected: true } socket)
            return FailedDispatch(destination, messages.Length, "OSC UDP socket is not connected.");

        try
        {
            foreach (var message in messages)
            {
                await socket.SendAsync(message.ToByteArray(), SocketFlags.None, ct);
            }

            OnMessagesDispatched(messages.Length);
            return SuccessfulDispatch(destination, messages.Length);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error sending OSC bundle");
            return FailedDispatch(destination, messages.Length, ex.Message);
        }
    }

    private static OscDispatchResult SuccessfulDispatch(string destination, int count) =>
        new(true, destination, DateTimeOffset.UtcNow, count, null);

    private static OscDispatchResult FailedDispatch(string destination, int count, string error) =>
        new(false, destination, DateTimeOffset.UtcNow, count, error);
}
