const std = @import("std");
const lz4 = @import("lz4");

pub const Roblox = @import("roblox.zig");
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
pub fn deserialize(allocator: std.mem.Allocator, contents: []const u8) !Roblox.Document {
    if (contents.len < 14)
        return Roblox.DeserializeError.UnknownEncoding;

    const encoding = EncodingMap.get(contents[0..14]) orelse EncodingMap.get(contents[0..7]) orelse return Roblox.DeserializeError.UnknownEncoding;

    switch (encoding) {
        .Binary => return try Binary.deserialize(allocator, contents[14..]),
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

    const contents = try file.reader().readAllAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(contents);

    var doc = try deserialize(allocator, contents);
    defer doc.deinit();

    const slice = try serialize(doc, .Binary);
    defer allocator.free(slice);

    var doc2 = try deserialize(allocator, slice);
    defer doc2.deinit();
}

test "XML" {
    // const allocator = std.testing.allocator;

    // const file = try getTestFile("places/baseplate-566/xml.rbxlx");

    // _ = try deserializePlace(allocator, file.reader().any());
}
