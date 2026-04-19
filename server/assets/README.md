# server/assets

Files embedded into the Go binary via `//go:embed assets/*`.

`install.sh` and `install.ps1` here are **copies** of the canonical scripts
under `scripts/install/`.  Go's `go:embed` refuses to walk outside the
package directory, so we keep a copy here.

When you edit the canonical scripts, re-sync with:

    cp scripts/install/install.sh  server/assets/install.sh
    cp scripts/install/install.ps1 server/assets/install.ps1

CI does the same copy before `go build` so the embedded copy is always
current in release artifacts.
