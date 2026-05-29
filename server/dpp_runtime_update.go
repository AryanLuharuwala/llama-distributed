package main

// dpp_runtime_update.go — auto-update for the Python diffusion runtime.
//
// Why this exists: as we add new BackboneAdapters (every time a new
// diffusion model family ships), only python/dpp_runtime/backbones.py
// changes — yet the gpunet-node binary on every rig embeds a copy of
// that directory.  Re-rolling the binary and asking every operator to
// redeploy is painful.  This endpoint lets the node fetch the latest
// runtime bundle on its own and swap it in atomically, with SHA256
// verification on every file so a MitM can't ship malicious .py.
//
// Wire shape:
//
//   GET /api/dpp/runtime/manifest
//     → { "version":   <unix-ts of bundle>,
//         "files":     [{ "path": "worker.py", "sha256": "…", "size": … }, …],
//         "tarball_sha256": "…",
//         "signature":     "<ed25519 sig of tarball_sha256>"  (optional) }
//
//   GET /api/dpp/runtime/bundle.tar.gz
//     → tar.gz of the runtime directory, in the same byte order as the
//       manifest was hashed against.
//
// The node side: at startup (and every N minutes), pull the manifest,
// compare tarball_sha256 to the cached value, download bundle if
// different, verify every file's sha against the manifest, then
// extract into a tmpdir and atomically rename over the live runtime
// dir.  Workers are restarted on the next process spawn.
//
// Signature gate: when DPP_RUNTIME_SIGN_KEY is set in the server env,
// the manifest is signed; the node refuses an unsigned manifest in
// strict mode.  For dev (no key), the signature field is omitted and
// the node falls back to TLS auth on the server.

import (
	"archive/tar"
	"compress/gzip"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// dppRuntimeFile describes one file inside the runtime bundle.
type dppRuntimeFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// dppRuntimeManifest is the JSON body the node fetches.  Field names
// match the wire-shape comment above so the node's struct decoder
// stays in lock-step.
type dppRuntimeManifest struct {
	Version       int64            `json:"version"`
	Files         []dppRuntimeFile `json:"files"`
	TarballSHA256 string           `json:"tarball_sha256"`
	Signature     string           `json:"signature,omitempty"`
}

// dppRuntimeBundle holds the in-memory tarball + computed manifest
// for the runtime directory.  Built once at startup; rebuilt at most
// once every 5 minutes if the on-disk runtime changes (dev workflow).
type dppRuntimeBundle struct {
	mu        sync.Mutex
	dir       string
	manifest  dppRuntimeManifest
	tarball   []byte
	builtAt   time.Time
	signKey   ed25519.PrivateKey // nil if unsigned
}

// loadDPPRuntimeBundle constructs a bundle from the on-disk runtime
// directory.  Returns nil + a warning if the directory doesn't
// exist — the server still runs, just without the auto-update
// endpoints serving a useful response.
func loadDPPRuntimeBundle() *dppRuntimeBundle {
	dir := os.Getenv("DPP_RUNTIME_DIR")
	if dir == "" {
		dir = "python/dpp_runtime"
	}
	if _, err := os.Stat(dir); err != nil {
		return nil
	}
	b := &dppRuntimeBundle{dir: dir}
	if keyB64 := os.Getenv("DPP_RUNTIME_SIGN_KEY"); keyB64 != "" {
		if raw, err := base64.StdEncoding.DecodeString(keyB64); err == nil && len(raw) == ed25519.PrivateKeySize {
			b.signKey = ed25519.PrivateKey(raw)
		}
	}
	if err := b.rebuild(); err != nil {
		return nil
	}
	return b
}

// rebuild walks dir, hashes every .py file, packs a tar.gz, and
// computes the manifest.  Holds b.mu for the duration.  Skips
// __pycache__ and test_* files to keep the bundle small.
func (b *dppRuntimeBundle) rebuild() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	files := []dppRuntimeFile{}
	var tarbuf strings.Builder
	gw := gzip.NewWriter(struct{ io.Writer }{&tarbufWriter{&tarbuf}})
	tw := tar.NewWriter(gw)
	defer tw.Close()
	defer gw.Close()

	err := filepath.Walk(b.dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if info.Name() == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}
		base := info.Name()
		if strings.HasPrefix(base, "test_") || strings.HasSuffix(base, ".pyc") {
			return nil
		}
		if !strings.HasSuffix(base, ".py") {
			return nil
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(b.dir, p)
		if err != nil {
			rel = base
		}
		sum := sha256.Sum256(raw)
		files = append(files, dppRuntimeFile{
			Path:   filepath.ToSlash(rel),
			SHA256: hex.EncodeToString(sum[:]),
			Size:   info.Size(),
		})
		hdr := &tar.Header{
			Name:    filepath.ToSlash(rel),
			Mode:    0o644,
			Size:    info.Size(),
			ModTime: info.ModTime(),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		if _, err := tw.Write(raw); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return err
	}
	_ = tw.Close()
	_ = gw.Close()
	// Stable manifest order — node verifies in the same order.
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })

	tarbytes := []byte(tarbuf.String())
	tarSum := sha256.Sum256(tarbytes)
	mf := dppRuntimeManifest{
		Version:       time.Now().Unix(),
		Files:         files,
		TarballSHA256: hex.EncodeToString(tarSum[:]),
	}
	if b.signKey != nil {
		sig := ed25519.Sign(b.signKey, tarSum[:])
		mf.Signature = base64.StdEncoding.EncodeToString(sig)
	}
	b.manifest = mf
	b.tarball = tarbytes
	b.builtAt = time.Now()
	return nil
}

// maybeRebuild rebuilds at most every 5 minutes.  Cheap on idle.
func (b *dppRuntimeBundle) maybeRebuild() {
	b.mu.Lock()
	stale := time.Since(b.builtAt) > 5*time.Minute
	b.mu.Unlock()
	if stale {
		_ = b.rebuild()
	}
}

func (s *server) handleDPPRuntimeManifest(w http.ResponseWriter, r *http.Request) {
	if s.dppRuntime == nil {
		writeErr(w, 404, "dpp runtime bundle not configured")
		return
	}
	s.dppRuntime.maybeRebuild()
	s.dppRuntime.mu.Lock()
	mf := s.dppRuntime.manifest
	s.dppRuntime.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(mf)
}

func (s *server) handleDPPRuntimeBundle(w http.ResponseWriter, r *http.Request) {
	if s.dppRuntime == nil {
		writeErr(w, 404, "dpp runtime bundle not configured")
		return
	}
	s.dppRuntime.maybeRebuild()
	s.dppRuntime.mu.Lock()
	tarball := s.dppRuntime.tarball
	sha := s.dppRuntime.manifest.TarballSHA256
	s.dppRuntime.mu.Unlock()
	w.Header().Set("Content-Type", "application/gzip")
	w.Header().Set("X-Bundle-SHA256", sha)
	w.Header().Set("Content-Disposition", `attachment; filename="dpp_runtime.tar.gz"`)
	_, _ = w.Write(tarball)
}

// tarbufWriter adapts strings.Builder to io.Writer so we can stream
// the tar+gzip output directly into the builder without an extra
// bytes.Buffer.  strings.Builder.Write exists but lives in a method
// set that gzip.NewWriter doesn't see through the interface struct
// shim above — wrapping is the simplest fix.
type tarbufWriter struct {
	sb *strings.Builder
}

func (w *tarbufWriter) Write(p []byte) (int, error) {
	return w.sb.Write(p)
}
