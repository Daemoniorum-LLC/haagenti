fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use bundled protoc from protobuf-src
    std::env::set_var("PROTOC", protobuf_src::protoc());

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/compression.proto"], &["proto"])?;
    Ok(())
}
