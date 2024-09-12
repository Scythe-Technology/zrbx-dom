const std = @import("std");

pub fn toBase64(allocator: std.mem.Allocator, buf: []const u8) ![]const u8 {
    const out = try allocator.alloc(u8, std.base64.standard.Encoder.calcSize(buf.len));
    defer allocator.free(out);
    const slice = std.base64.standard.Encoder.encode(out, buf);
    return try allocator.dupe(u8, slice);
}

pub fn fromBase64(allocator: std.mem.Allocator, buf: []const u8) ![]const u8 {
    const out = try allocator.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(buf));
    errdefer allocator.free(out);
    try std.base64.standard.Decoder.decode(out, buf);
    return out;
}

test "toBase64" {
    const allocator = std.testing.allocator;
    const base64 = try toBase64(allocator, "Some String Value");
    defer allocator.free(base64);
    try std.testing.expectEqualStrings("U29tZSBTdHJpbmcgVmFsdWU=", base64);
}

test "fromBase64" {
    const allocator = std.testing.allocator;
    const string = try fromBase64(allocator, "U29tZSBTdHJpbmcgVmFsdWU=");
    defer allocator.free(string);
    try std.testing.expectEqualStrings("Some String Value", string);
}
