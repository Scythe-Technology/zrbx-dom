const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dep_lz4 = b.dependency("lz4", .{ .target = target, .optimize = optimize });

    const module = b.addModule("zrbx-dom", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    module.addImport("lz4", dep_lz4.module("zig-lz4"));

    const tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // tests.root_module.addImport("zrbx-dom", module);
    tests.root_module.addImport("lz4", dep_lz4.module("zig-lz4"));

    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
