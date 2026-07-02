namespace BabblePersonalizer.Core.Onnx;

/// <summary>Reads only the default-domain opset from ModelProto; ONNX Runtime performs full validation.</summary>
internal static class OnnxProtobufHeaderReader
{
    public static long? ReadDefaultOpset(string path)
    {
        using var stream = File.OpenRead(path);
        while (stream.Position < stream.Length)
        {
            var key = ReadVarint(stream);
            var field = (int)(key >> 3);
            var wire = (int)(key & 7);
            if (field == 8 && wire == 2)
            {
                var bytes = new byte[checked((int)ReadVarint(stream))];
                stream.ReadExactly(bytes);
                var result = ParseOpset(bytes);
                if (result.Domain is "" or "ai.onnx") return result.Version;
            }
            else Skip(stream, wire);
        }
        return null;
    }

    private static (string Domain, long Version) ParseOpset(byte[] bytes)
    {
        using var stream = new MemoryStream(bytes, false);
        var domain = ""; long version = 0;
        while (stream.Position < stream.Length)
        {
            var key = ReadVarint(stream); var field = (int)(key >> 3); var wire = (int)(key & 7);
            if (field == 1 && wire == 2)
            {
                var value = new byte[checked((int)ReadVarint(stream))]; stream.ReadExactly(value);
                domain = System.Text.Encoding.UTF8.GetString(value);
            }
            else if (field == 2 && wire == 0) version = (long)ReadVarint(stream);
            else Skip(stream, wire);
        }
        return (domain, version);
    }

    private static ulong ReadVarint(Stream stream)
    {
        ulong result = 0; var shift = 0;
        while (shift < 64)
        {
            var next = stream.ReadByte();
            if (next < 0) throw new EndOfStreamException();
            result |= (ulong)(next & 0x7f) << shift;
            if ((next & 0x80) == 0) return result;
            shift += 7;
        }
        throw new InvalidDataException("Invalid protobuf varint.");
    }

    private static void Skip(Stream stream, int wireType)
    {
        switch (wireType)
        {
            case 0: ReadVarint(stream); break;
            case 1: stream.Seek(8, SeekOrigin.Current); break;
            case 2: stream.Seek(checked((long)ReadVarint(stream)), SeekOrigin.Current); break;
            case 5: stream.Seek(4, SeekOrigin.Current); break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
    }
}
