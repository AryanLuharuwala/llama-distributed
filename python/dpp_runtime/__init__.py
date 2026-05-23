# Diffusion pipeline-parallel runtime for llama-distributed.
#
# One process per dpp_route the agent receives.  The process owns its
# assigned slice of a diffusion model (text encoders, a UNet block range,
# or VAE) and exchanges ACTV-shaped frames with the C++ agent over a
# loopback TCP socket.  The agent in turn shuttles frames to/from
# dist-server over its WS.
#
# We deliberately reimplement the wire format here (instead of importing
# the Go encoder) so the runtime has no Go dependency at runtime.
