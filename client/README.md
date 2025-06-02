# OpenGL Shading Language for Visual Studio Code
This extension adds language support for the OpenGL Shading Language to Visual Studio Code.

## Note on this Fork
This is a fork of the original `glsl-lsp` extension by [kuba_p](https://github.com/KubaP). This version, published by [jsdf](https://github.com/jsdf), adds support for Apple Silicon, which was lacking in the previously published version. 

## original readme

It current supports the following features:
- Basic semantic highlighting
- Syntax errors
- Limited semantic diagnostics
- Macro expansion
- Conditional compilation evaluation and control

⚠ Most features are currently unimplemented; this includes name resolution.

Currently supported versions:
|GLSL version|OpenGL version|
|-|-|
|460|4.6|
|450|4.5|

<!-- How does this extension compare to existing ones? -->

## Security & Privacy
This extension, and the `glsl-lsp` server, do not execute arbitrary source code; any evaluations are free of any side-effects. Furthermore, currently no I/O or network calls are made.

No telemetry is collected by this extension.

## Installation
Simply install this extension and you're good to go! The platform-specific server binary is packaged along with all the other files, so there's no need to download extra stuff through a separate package manager.

## Troubleshooting
This extension is in a preview state and it has not been fully tested. Crashes may occur. 

Start troubleshooting by looking at the server logs. Log messages are found in the `Output > GLSL Language Server` channel. If you see a backtrace, that means the server has crashed. Log messages from this extension's language client are found in the `Output > GLSL Client` channel.

To fully capture the messages between the editor and the server, set the `glsl.trace.server` setting to `"verbose"`. These messages will be logged to the `Output > GLSL Language Server Trace` channel.

This fork is provided as-is. You can get the current version information by running `GLSL: Show version` from the command palette.

## Contribution
<!-- Contributions are always welcome, be it code, tests, documentation or bug reports, feature requests, etc. Please see the [contribution guide]() for more details. For help, please file an issue in the repo. -->
The original project by [kuba_p](https://github.com/KubaP) may still be open to contributions.

## License
This project is licensed under the **MIT** license - see [LICENSE](LICENSE) for details.
