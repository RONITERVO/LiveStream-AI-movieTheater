# Security

This application is designed to bind to `127.0.0.1` and control trusted local AI services. Do not expose the UI, ComfyUI, llama.cpp, or Supertonic ports directly to an untrusted network.

The project contains no authentication layer. If remote access is required, place an authenticated reverse proxy in front of every exposed service and review the media-file routes first.

Report security issues privately to the repository owner rather than posting exploit details in a public issue.
