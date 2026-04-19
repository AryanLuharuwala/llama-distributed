# llama-distributed — install

This archive contains the `dist-node`, `dist-join`, `dist-client`, and
`dist-coordinator` binaries for your platform, plus installer scripts that
register the node agent as a background service.

## Quick start

Grab a pool-join token from your coordinator's web dashboard, then:

**Linux**
```
./scripts/install/linux-install.sh install --token <POOL_TOKEN>
```

**macOS**
```
./scripts/install/macos-install.sh install --token <POOL_TOKEN>
```

**Windows (admin PowerShell)**
```
powershell -ExecutionPolicy Bypass -File scripts\install\windows-install.ps1 -Token <POOL_TOKEN>
```

Each installer creates a user-scoped service that auto-starts, re-pairs with
the coordinator, and restarts on failure.

## Uninstall

Run the same script with `uninstall` (Linux/macOS) or `-Uninstall` (Windows).

## Manual run (no service)

```
./bin/dist-join --token <POOL_TOKEN> --server pool.llamadist.dev -g 999
```

`-g 999` offloads every layer it can onto the GPU. Drop to `-g 0` for CPU-only.
