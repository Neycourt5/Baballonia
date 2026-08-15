using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.Sockets;
using VRCFaceTracking.Core.OSC;

namespace VRCFaceTracking.Baballonia;

public class BabbleOsc
{
    public static readonly float[] EyeExpressions = new float[12];

    private Socket? _receiver;

    private bool _loop = true;

    private readonly Thread? _thread;

    private readonly int _resolvedPort;

    private readonly string? _resolvedHost;

    private const string DefaultHost = "127.0.0.1";

    private const int DefaultPort = 8888;

    private const int TimeoutMs = 10000;

    public BabbleOsc(ILogger iLogger, string host, int? port)
    {
        if (_receiver != null)
        {
            iLogger.LogError("BabbleEyeOSC connection already exists.");
            return;
        }
        _resolvedHost = host ?? DefaultHost;
        _resolvedPort = port ?? DefaultPort;

        iLogger.LogInformation($"Started BabbleEyeOSC with Host: {_resolvedHost} and Port {_resolvedPort}");
        ConfigureReceiver();
        _loop = true;
        _thread = new Thread(ListenLoop);
        _thread.Start();
    }

    private void ConfigureReceiver()
    {
        IPAddress address = IPAddress.Parse(_resolvedHost!);
        IPEndPoint localEp = new IPEndPoint(address, _resolvedPort);
        _receiver = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        _receiver.Bind(localEp);
        _receiver.ReceiveTimeout = TimeoutMs;
    }

    private void ListenLoop()
    {
        byte[] array = new byte[4096];
        while (_loop)
        {
            try
            {
                if (_receiver!.IsBound)
                {
                    int len = _receiver.Receive(array);
                    int messageIndex = 0;
                    OscMessage oscMessage;
                    try
                    {
                        oscMessage = new OscMessage(array, len, ref messageIndex);
                    }
                    catch (Exception)
                    {
                        continue;
                    }
                    if (oscMessage.Value is float value)
                    {
                        if (EyeExpressionRouter.TryApply(oscMessage.Address, value, EyeExpressions))
                            continue;

                        if (string.Equals(oscMessage.Address, "/LeftEyeBrow", StringComparison.OrdinalIgnoreCase))
                        {
                            EyeExpressions[(int)ExpressionMapping.EyeLeftLower] = value;
                        }
                        else if (string.Equals(
                                     oscMessage.Address,
                                     "/RightEyeBrow",
                                     StringComparison.OrdinalIgnoreCase))
                        {
                            EyeExpressions[(int)ExpressionMapping.EyeRightLower] = value;
                        }
                        else if (BabbleExpressions.BabbleExpressionMap.ContainsKey2(oscMessage.Address))
                        {
                            BabbleExpressions.BabbleExpressionMap.SetByKey2(oscMessage.Address, value);
                        }
                    }
                }
                else
                {
                    _receiver.Close();
                    _receiver.Dispose();
                    ConfigureReceiver();
                }
            }
            catch (Exception)
            {
                // ignore
            }
        }
    }

    public void Teardown()
    {
        _loop = false;
        _receiver!.Close();
        _receiver.Dispose();
        _thread!.Join();
    }
}
