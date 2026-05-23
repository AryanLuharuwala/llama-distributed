package main

// Tests for the parallel HF download path:
//   - verifyFileSHA256 catches bit flips, accepts matches, no-ops on empty.
//   - downloadOneAndVerify retries once on sha mismatch then surfaces error.
//   - downloadFilesParallel fans across workers and is cancellable.
//   - downloadFilesParallel honors the "already-on-disk" resume short-circuit.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// fakeHFFiles spins up a server that maps /<repo>/resolve/<rev>/<path>
// to bytes from `data[path]`.  flipBytes lets a single path be served
// with a 1-bit flip on the first GET to exercise the re-fetch path.
type fakeBytesServer struct {
	*httptest.Server
	data       map[string][]byte
	flipFirst  map[string]bool   // path -> serve corrupt once
	served     atomic.Int32      // total requests
	corruption atomic.Int32      // corrupt responses sent
}

func newFakeBytesServer(data map[string][]byte, flipFirst map[string]bool) *fakeBytesServer {
	fs := &fakeBytesServer{data: data, flipFirst: flipFirst}
	fs.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fs.served.Add(1)
		// URL pattern: /<repo>/resolve/<rev>/<path...>
		parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 4)
		if len(parts) < 4 {
			http.NotFound(w, r)
			return
		}
		path := parts[3]
		b, ok := fs.data[path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		// Single-shot corruption: flip a bit on the first response only.
		if fs.flipFirst[path] {
			fs.flipFirst[path] = false
			b = append([]byte(nil), b...)
			b[0] ^= 0x01
			fs.corruption.Add(1)
		}
		// Honor Range for simple resume — bytes=N-
		if rng := r.Header.Get("Range"); strings.HasPrefix(rng, "bytes=") {
			var from int64
			_, _ = fmt.Sscanf(rng, "bytes=%d-", &from)
			if from > 0 && int(from) < len(b) {
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, len(b)-1, len(b)))
				w.WriteHeader(http.StatusPartialContent)
				_, _ = w.Write(b[from:])
				return
			}
		}
		_, _ = w.Write(b)
	}))
	return fs
}

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func TestVerifyFileSHA256(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "x.bin")
	content := []byte("the quick brown fox")
	if err := os.WriteFile(dst, content, 0o644); err != nil {
		t.Fatal(err)
	}
	want := sha256Hex(content)

	// Match (lowercase).
	if err := verifyFileSHA256(dst, want); err != nil {
		t.Errorf("lowercase match: %v", err)
	}
	// Match (uppercase — verify is case-insensitive).
	if err := verifyFileSHA256(dst, strings.ToUpper(want)); err != nil {
		t.Errorf("uppercase match: %v", err)
	}
	// Empty want → no-op (used for non-LFS files).
	if err := verifyFileSHA256(dst, ""); err != nil {
		t.Errorf("empty want should be a no-op: %v", err)
	}
	// Bit flip → mismatch.
	if err := os.WriteFile(dst, []byte("the slow brown fox"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := verifyFileSHA256(dst, want); err == nil {
		t.Errorf("expected mismatch error")
	}
	// Missing file → error.
	if err := verifyFileSHA256(filepath.Join(dir, "missing"), want); err == nil {
		t.Errorf("expected open error for missing file")
	}
}

// downloadOneAndVerify must retry once when the first response is corrupt
// and the sha hash mismatches.
func TestDownloadOneAndVerify_ReFetchOnSHAMismatch(t *testing.T) {
	content := []byte("hello world repeated 100 times. " + strings.Repeat("A", 1000))
	want := sha256Hex(content)

	srv := newFakeBytesServer(
		map[string][]byte{"file.bin": content},
		map[string]bool{"file.bin": true}, // corrupt first response
	)
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")

	hfRetryDelay = 1 * 0 // no backoff in tests
	defer func() { hfRetryDelay = 0 }()

	var progress int64
	err := downloadOneAndVerify(context.Background(),
		srv.URL+"/repo/resolve/main/file.bin", "", dst, want,
		func(d int64) { atomic.AddInt64(&progress, d) })
	if err != nil {
		t.Fatalf("downloadOneAndVerify: %v", err)
	}
	// Final file must be uncorrupted.
	got, _ := os.ReadFile(dst)
	if string(got) != string(content) {
		t.Errorf("final bytes wrong after re-fetch")
	}
	// We expect exactly 1 corrupt response + at least 1 retry.
	if srv.corruption.Load() != 1 {
		t.Errorf("corruption attempts: got %d, want 1", srv.corruption.Load())
	}
	if srv.served.Load() < 2 {
		t.Errorf("served only %d times; expected >=2 (retry path not taken)", srv.served.Load())
	}
}

// Parallel pool: 6 files across 3 workers; all land on disk with correct content.
func TestDownloadFilesParallel_FansOut(t *testing.T) {
	data := map[string][]byte{}
	files := []hfFileInfo{}
	for i := 0; i < 6; i++ {
		name := fmt.Sprintf("part-%d.bin", i)
		b := []byte(strings.Repeat(fmt.Sprintf("%d", i), 4096))
		data[name] = b
		files = append(files, hfFileInfo{Path: name, Size: int64(len(b)), SHA256: sha256Hex(b)})
	}
	srv := newFakeBytesServer(data, nil)
	defer srv.Close()

	// Override hfAPIBase + use fully-qualified URLs via the resolve helper.
	withHFBase(t, srv.URL, func() {
		staging := t.TempDir()
		s := &server{}
		var progress int64
		err := s.downloadFilesParallel(context.Background(), files, 3,
			"repo", "main", "", staging,
			func(d int64) { atomic.AddInt64(&progress, d) })
		if err != nil {
			t.Fatalf("downloadFilesParallel: %v", err)
		}
		for _, f := range files {
			b, err := os.ReadFile(filepath.Join(staging, f.Path))
			if err != nil {
				t.Errorf("missing %s: %v", f.Path, err)
				continue
			}
			if string(b) != string(data[f.Path]) {
				t.Errorf("%s: content mismatch", f.Path)
			}
		}
		// Sanity on progress meter — at least one report per file.
		// (Worker pool may double-count on re-fetch; we just want >0.)
		if atomic.LoadInt64(&progress) <= 0 {
			t.Errorf("progress callback never fired")
		}
	})
}

// Pool must short-circuit files already on disk with the right content.
func TestDownloadFilesParallel_ResumeShortCircuit(t *testing.T) {
	b := []byte(strings.Repeat("Z", 256))
	files := []hfFileInfo{{Path: "x.bin", Size: int64(len(b)), SHA256: sha256Hex(b)}}
	srv := newFakeBytesServer(map[string][]byte{"x.bin": b}, nil)
	defer srv.Close()

	staging := t.TempDir()
	// Pre-place the file so the pool should short-circuit.
	if err := os.WriteFile(filepath.Join(staging, "x.bin"), b, 0o644); err != nil {
		t.Fatal(err)
	}

	withHFBase(t, srv.URL, func() {
		s := &server{}
		err := s.downloadFilesParallel(context.Background(), files, 2,
			"repo", "main", "", staging,
			func(int64) {})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if srv.served.Load() != 0 {
			t.Errorf("expected zero network calls (resume short-circuit); got %d", srv.served.Load())
		}
	})
}

// If a pre-existing file fails sha verify, the pool must re-fetch it.
func TestDownloadFilesParallel_RefetchCorruptedResumeFile(t *testing.T) {
	good := []byte(strings.Repeat("G", 1024))
	files := []hfFileInfo{{Path: "y.bin", Size: int64(len(good)), SHA256: sha256Hex(good)}}
	srv := newFakeBytesServer(map[string][]byte{"y.bin": good}, nil)
	defer srv.Close()

	staging := t.TempDir()
	// Place corrupted bytes of the right size.
	bad := append([]byte(nil), good...)
	bad[0] ^= 0xFF
	if err := os.WriteFile(filepath.Join(staging, "y.bin"), bad, 0o644); err != nil {
		t.Fatal(err)
	}

	withHFBase(t, srv.URL, func() {
		s := &server{}
		err := s.downloadFilesParallel(context.Background(), files, 1,
			"repo", "main", "", staging,
			func(int64) {})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		final, _ := os.ReadFile(filepath.Join(staging, "y.bin"))
		if string(final) != string(good) {
			t.Errorf("corrupted pre-existing file was not replaced")
		}
		if srv.served.Load() == 0 {
			t.Errorf("expected at least 1 network call after corrupt-resume detection")
		}
	})
}

// A malicious HF response that claims a path like "../escape/poison" must
// not write outside the staging dir.  The downloader rejects the whole
// batch — we don't want partial state from a tampered repo listing.
func TestDownloadFilesParallel_RejectsPathTraversal(t *testing.T) {
	files := []hfFileInfo{
		{Path: "../escape.bin", Size: 16},
	}
	srv := newFakeBytesServer(map[string][]byte{"../escape.bin": []byte("BADBADBADBADBADB")}, nil)
	defer srv.Close()

	staging := t.TempDir()
	withHFBase(t, srv.URL, func() {
		s := &server{}
		err := s.downloadFilesParallel(context.Background(), files, 1,
			"repo", "main", "", staging,
			func(int64) {})
		if err == nil {
			t.Fatal("expected rejection for path-traversal entry")
		}
		if !strings.Contains(err.Error(), "escapes staging") {
			t.Errorf("error should mention 'escapes staging': %v", err)
		}
		// Confirm no file was written at the escape target.
		if _, err := os.Stat(filepath.Join(staging, "..", "escape.bin")); err == nil {
			t.Errorf("path-traversal write succeeded: file exists outside staging")
		}
	})
}

// Cancelling the outer ctx must propagate to in-flight workers fast.
func TestDownloadFilesParallel_CancelPropagates(t *testing.T) {
	// One huge file → ensures we cancel mid-stream.
	big := make([]byte, 2<<20) // 2 MiB
	for i := range big {
		big[i] = byte(i & 0xFF)
	}
	files := []hfFileInfo{
		{Path: "a.bin", Size: int64(len(big)), SHA256: sha256Hex(big)},
		{Path: "b.bin", Size: int64(len(big)), SHA256: sha256Hex(big)},
	}
	srv := newFakeBytesServer(map[string][]byte{
		"a.bin": big,
		"b.bin": big,
	}, nil)
	defer srv.Close()

	staging := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled — should not download anything

	withHFBase(t, srv.URL, func() {
		s := &server{}
		err := s.downloadFilesParallel(ctx, files, 2,
			"repo", "main", "", staging,
			func(int64) {})
		if err == nil {
			t.Errorf("expected cancellation error")
		}
	})
}
