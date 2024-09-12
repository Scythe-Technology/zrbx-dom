const std = @import("std");
const lz4 = @import("lz4");

const Roblox = @import("roblox.zig");
const Binary = @import("./formats/binary.zig");

const Dom = struct {};

const Encoding = enum {
    Binary,
    Xml,
};

const EncodingMap = std.StaticStringMap(Encoding).initComptime(.{
    .{ Binary.BinaryMagicHeader, .Binary },
    .{ "<roblox", .Xml },
});

pub fn serialize(doc: Roblox.Document, encoding: Encoding) ![]const u8 {
    switch (encoding) {
        .Binary => {
            return try Binary.serialize(
                doc.allocator,
                doc.instances,
                doc.info,
                doc.metadata,
                doc.sharedstrings,
                doc.signatures,
            );
        },
        else => {},
    }
    return "NOTHING";
}

pub fn deserialize(allocator: std.mem.Allocator, reader: anytype) !Roblox.Document {
    const compressed = try lz4.Standard.compress(std.testing.allocator, "test");
    defer std.testing.allocator.free(compressed);

    var data: [14]u8 = undefined;
    const amount = try reader.read(&data);
    if (amount < 14) return Roblox.DeserializeError.UnknownEncoding;

    const encoding = EncodingMap.get(&data) orelse EncodingMap.get(data[0..7]) orelse return Roblox.DeserializeError.UnknownEncoding;

    switch (encoding) {
        .Binary => return try Binary.deserialize(allocator, reader),
        .Xml => {
            std.debug.print("XML\n", .{});
        },
    }
    return Roblox.DeserializeError.BadEncoding;
}

test {
    _ = Binary;
}

const RBX_TEST_FILES = "rbx-test-files";
pub fn getTestFile(path: []const u8) !std.fs.File {
    const cwd = std.fs.cwd();
    const test_files = try cwd.openDir(RBX_TEST_FILES, .{});
    return try test_files.openFile(path, .{});
}

test "Binary" {
    const allocator = std.testing.allocator;

    const file = try getTestFile("places/baseplate-566/binary.rbxl");

    var doc = try deserialize(allocator, file.reader());
    defer doc.deinit();

    const slice = try serialize(doc, .Binary);
    defer allocator.free(slice);

    var sample = std.io.fixedBufferStream(slice);

    var doc2 = try deserialize(allocator, sample.reader());
    defer doc2.deinit();
}

test "XML" {
    // const allocator = std.testing.allocator;

    // const file = try getTestFile("places/baseplate-566/xml.rbxlx");

    // _ = try deserializePlace(allocator, file.reader().any());
}
