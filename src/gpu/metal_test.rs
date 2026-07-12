#[cfg(feature = "gpu-metal")]
use metal::*;
#[cfg(feature = "gpu-metal")]
fn test(device: &Device, command_buffer: &CommandBufferRef) {
    let event = device.new_shared_event();
    command_buffer.encode_signal_event(&event, 1);
    let val = event.signaled_value();
}
