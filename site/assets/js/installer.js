/* installer.js — drives the 5-axis install picker and generates the command */

/**
 * @typedef {{ build: string, os: string, role: string, accel: string, method: string }} PickerState
 * @typedef {{ id: string, label: string }} PickerOption
 * @typedef {{ key: string, label: string, opts: PickerOption[] }} PickerAxis
 * @typedef {{ kind: "warn" | "info", text: string }} PickerNote
 */

(function () {
  "use strict";

  /** @type {PickerState} */
  var state = {
    build: "stable",
    os: "linux",
    role: "rig",
    accel: "cuda",
    method: "oneliner"
  };

  /** @type {PickerAxis[]} */
  var AXES = [
    { key: "build",  label: "Build",       opts: [
      { id: "stable",  label: "Stable" },
      { id: "nightly", label: "Nightly" }
    ]},
    { key: "os",     label: "OS",          opts: [
      { id: "linux",   label: "Linux" },
      { id: "macos",   label: "macOS" },
      { id: "windows", label: "Windows" }
    ]},
    { key: "role",   label: "Role",        opts: [
      { id: "rig",     label: "Rig (agent)" },
      { id: "server",  label: "Server (control plane)" },
      { id: "client",  label: "Client" }
    ]},
    { key: "accel",  label: "Accelerator", opts: [
      { id: "cuda",   label: "CUDA" },
      { id: "metal",  label: "Metal" },
      { id: "vulkan", label: "Vulkan" },
      { id: "cpu",    label: "CPU" }
    ]},
    { key: "method", label: "Method",      opts: [
      { id: "oneliner", label: "One-liner" },
      { id: "tarball",  label: "Prebuilt tarball" },
      { id: "source",   label: "Build from source" }
    ]}
  ];

  var REPO = "AryanLuharuwala/llama-distributed";

  /**
   * @param {string} os
   * @param {string} accel
   * @returns {boolean}
   */
  function combosValid(os, accel) {
    if (os === "linux")   return accel === "cuda"   || accel === "vulkan" || accel === "cpu";
    if (os === "macos")   return accel === "metal"  || accel === "cpu";
    if (os === "windows") return accel === "cuda"   || accel === "vulkan" || accel === "cpu";
    return false;
  }

  function coerce() {
    if (!combosValid(state.os, state.accel)) {
      state.accel = state.os === "macos" ? "metal" : "cuda";
    }
  }

  /** @returns {string} */
  function generate() {
    if (state.role === "server") return serverCmd();
    if (state.role === "client") return clientCmd();
    if (state.method === "source")  return buildSourceCmd(state.os, state.accel, "rig");
    if (state.method === "tarball") return tarballCmd(state.os, state.accel);
    return oneLinerCmd(state.os, state.build, state.accel);
  }

  /** @returns {string} */
  function serverCmd() {
    if (state.method === "source") return buildSourceCmd(state.os, state.accel, "server");
    if (state.os === "windows") {
      return [
        "# Windows PowerShell (admin)",
        "$url = \"https://github.com/" + REPO + "/releases/latest/download/dist-server-windows-x86_64.zip\"",
        "Invoke-WebRequest -Uri $url -OutFile dist-server.zip",
        "Expand-Archive dist-server.zip -DestinationPath $env:LOCALAPPDATA\\llama-distributed",
        "& $env:LOCALAPPDATA\\llama-distributed\\dist-server.exe -dev"
      ].join("\n");
    }
    var suffix = state.os === "macos" ? "macos" : "linux";
    return [
      "# Run the control plane (dashboard + OpenAI-compat endpoints)",
      "curl -fsSL https://github.com/" + REPO + "/releases/latest/download/dist-server-" + suffix + "-x86_64.tar.gz | tar xz",
      "./dist-server -dev"
    ].join("\n");
  }

  /** @returns {string} */
  function clientCmd() {
    if (state.os === "windows") {
      return [
        "# Windows — download dist-client.exe and point it at your pool",
        "iwr https://github.com/" + REPO + "/releases/latest/download/dist-client-windows-x86_64.exe -OutFile dist-client.exe",
        ".\\dist-client.exe --server SERVER --prompt \"Hello!\""
      ].join("\n");
    }
    var suffix = state.os === "macos" ? "macos" : "linux";
    return [
      "# Download dist-client and send a prompt",
      "curl -fsSL https://github.com/" + REPO + "/releases/latest/download/dist-client-" + suffix + "-x86_64 -o dist-client",
      "chmod +x dist-client",
      "./dist-client --server SERVER --prompt \"Hello!\""
    ].join("\n");
  }

  /**
   * @param {string} os
   * @param {string} build
   * @param {string} accel
   * @returns {string}
   */
  function oneLinerCmd(os, build, accel) {
    var versionArg = build === "nightly" ? " --version nightly" : "";
    if (os === "windows") {
      return [
        "# Paste into an elevated PowerShell on your rig",
        "$pair = 'distpool://pair?token=TOKEN&server=wss://SERVER/ws/agent'",
        "iwr https://SERVER/install.ps1 -UseBasicParsing | iex"
      ].join("\n");
    }
    return [
      "# Paste on your rig — auto-detects " + accel.toUpperCase() + " and pairs with your server",
      "curl -fsSL https://SERVER/install.sh \\",
      "  | sh -s -- --pair 'distpool://pair?token=TOKEN&server=wss://SERVER/ws/agent'" + versionArg
    ].join("\n");
  }

  /**
   * @param {string} os
   * @param {string} accel
   * @returns {string}
   */
  function tarballCmd(os, accel) {
    var targetOs = os === "macos" ? "macos" : os;
    var arch = os === "macos" ? "arm64" : "x86_64";
    var target = targetOs + "-" + arch + "-" + accel;

    if (os === "windows") {
      return [
        "# Download prebuilt tarball for " + target,
        "$tag = (iwr https://api.github.com/repos/" + REPO + "/releases/latest | ConvertFrom-Json).tag_name",
        "$asset = \"llama-distributed-$tag-windows-x86_64-" + accel + ".zip\"",
        "iwr \"https://github.com/" + REPO + "/releases/download/$tag/$asset\" -OutFile $asset",
        "Expand-Archive $asset -DestinationPath $env:LOCALAPPDATA\\llama-distributed",
        "& $env:LOCALAPPDATA\\llama-distributed\\scripts\\install\\windows-install.ps1 -Pair 'distpool://...'"
      ].join("\n");
    }
    var platformScript = os === "macos" ? "macos" : "linux";
    return [
      "# Download prebuilt tarball for " + target,
      "tag=$(curl -fsSL https://api.github.com/repos/" + REPO + "/releases/latest | grep tag_name | cut -d\\\" -f4)",
      "asset=\"llama-distributed-${tag}-" + target + ".tar.gz\"",
      "curl -fSL \"https://github.com/" + REPO + "/releases/download/${tag}/${asset}\" -o \"${asset}\"",
      "tar xzf \"${asset}\"",
      "cd \"llama-distributed-${tag}-" + target + "\"",
      "./scripts/install/" + platformScript + "-install.sh install --pair 'distpool://...'"
    ].join("\n");
  }

  /**
   * @param {string} os
   * @param {string} accel
   * @param {string} role
   * @returns {string}
   */
  function buildSourceCmd(os, accel, role) {
    var flagStr;
    if      (accel === "cuda")   flagStr = " --cuda";
    else if (accel === "metal")  flagStr = " --metal";
    else if (accel === "vulkan") flagStr = " --vulkan";
    else                          flagStr = " # CPU build";

    if (os === "windows") {
      var psFlags = "";
      if (accel === "cuda")   psFlags = " -Cuda";
      if (accel === "vulkan") psFlags = " -Vulkan";
      return [
        "# Build from source on Windows (PowerShell + Visual Studio Build Tools)",
        "git clone --recursive https://github.com/" + REPO + ".git",
        "cd llama-distributed",
        ".\\scripts\\install\\build.ps1" + psFlags
      ].join("\n");
    }

    var run;
    if (role === "server")      run = "cd server && go build -o dist-server ./... && ./dist-server -dev";
    else if (role === "client") run = "./build/dist-client --server SERVER --prompt \"Hello!\"";
    else                        run = "./build/dist-node --pair 'distpool://pair?token=TOKEN&server=wss://SERVER/ws/agent'";

    return [
      "# Build from source — submodule init is automatic",
      "git clone --recursive https://github.com/" + REPO + ".git",
      "cd llama-distributed",
      "./scripts/build.sh" + flagStr,
      run
    ].join("\n");
  }

  /**
   * Tokenize a shell-ish line into styled spans, avoiding false positives
   * on angle-bracketed placeholders and environment-variable sigils.
   * @param {string} cmd
   * @returns {string}
   */
  function colorize(cmd) {
    return cmd.split("\n").map(/** @param {string} line */ function (line) {
      if (/^\s*#/.test(line))
        return '<span class="tok-cmt">' + escapeHtml(line) + "</span>";

      /** @type {{ kind: "flag" | "str" | "text", text: string }[]} */
      var segments = [];
      var i = 0;
      while (i < line.length) {
        var ch = line.charAt(i);

        if (ch === "'" || ch === '"') {
          var end = line.indexOf(ch, i + 1);
          if (end === -1) end = line.length - 1;
          segments.push({ kind: "str", text: line.slice(i, end + 1) });
          i = end + 1;
          continue;
        }

        if (ch === "-" && (i === 0 || /\s/.test(line.charAt(i - 1)))) {
          var m = /^--?[A-Za-z][A-Za-z0-9-]*/.exec(line.slice(i));
          if (m) {
            segments.push({ kind: "flag", text: m[0] });
            i += m[0].length;
            continue;
          }
        }

        var next = line.length;
        var q1 = line.indexOf("'", i);  if (q1 !== -1 && q1 < next) next = q1;
        var q2 = line.indexOf('"', i);  if (q2 !== -1 && q2 < next) next = q2;
        var flagMatch = line.slice(i).search(/\s-[A-Za-z-]/);
        if (flagMatch !== -1 && (i + flagMatch + 1) < next)
          next = i + flagMatch + 1;
        if (next === i) next = i + 1;
        segments.push({ kind: "text", text: line.slice(i, next) });
        i = next;
      }
      return segments.map(function (s) {
        var safe = escapeHtml(s.text);
        if (s.kind === "flag") return '<span class="tok-flag">' + safe + "</span>";
        if (s.kind === "str")  return '<span class="tok-str">'  + safe + "</span>";
        return safe;
      }).join("");
    }).join("\n");
  }

  /**
   * @param {string} s
   * @returns {string}
   */
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
  }

  /** @returns {string} */
  function prereq() {
    /** @type {string[]} */
    var parts = [];
    if (state.accel === "cuda")   parts.push("NVIDIA driver 535+ &middot; CUDA 12.5");
    if (state.accel === "metal")  parts.push("macOS 13+ on Apple Silicon");
    if (state.accel === "vulkan") parts.push("Vulkan 1.3 runtime");
    if (state.accel === "cpu")    parts.push("Any modern CPU (AVX2 recommended)");
    if (state.method === "source") {
      parts.push(state.os === "windows"
        ? "Visual Studio 2022 Build Tools &middot; CMake 3.14+"
        : "CMake 3.14+ &middot; GCC 11+ or Clang 14+");
    }
    if (state.role === "server") parts.push("Go 1.25+ &middot; SQLite3");
    return parts.join(" &middot; ");
  }

  /** @returns {PickerNote | null} */
  function note() {
    if (state.os === "macos" && (state.accel === "cuda" || state.accel === "vulkan"))
      return { kind: "warn", text: "CUDA/Vulkan not available on macOS — switch to Metal or CPU." };
    if (state.os === "linux" && state.accel === "metal")
      return { kind: "warn", text: "Metal is Apple-only — pick CUDA, Vulkan, or CPU on Linux." };
    if (state.role === "server" && state.accel !== "cpu")
      return { kind: "info", text: "The control plane is pure Go — the accelerator setting only affects compute rigs." };
    if (state.build === "nightly")
      return { kind: "info", text: "Nightly builds track the <code>main</code> branch. Expect occasional breakage." };
    if (state.method === "source")
      return { kind: "info", text: "Source builds take ~5 min for llama.cpp + ~1 min for our code." };
    return null;
  }

  /**
   * @param {HTMLElement} root
   */
  function render(root) {
    coerce();

    if (!root.querySelector(".picker-row")) {
      var bodyEl = /** @type {HTMLElement | null} */ (root.querySelector("[data-picker-body]"));
      if (!bodyEl) return;
      /** @type {HTMLElement} */
      var body = bodyEl;
      AXES.forEach(function (axis) {
        var row = document.createElement("div");
        row.className = "picker-row";
        row.dataset.axis = axis.key;

        var lbl = document.createElement("div");
        lbl.className = "picker-label";
        lbl.textContent = axis.label;
        row.appendChild(lbl);

        var opts = document.createElement("div");
        opts.className = "picker-opts";
        axis.opts.forEach(function (opt) {
          var btn = document.createElement("button");
          btn.type = "button";
          btn.className = "picker-opt";
          btn.dataset.axis = axis.key;
          btn.dataset.value = opt.id;
          btn.textContent = opt.label;
          btn.addEventListener("click", function () {
            /** @type {Record<string, string>} */ (state)[axis.key] = opt.id;
            render(root);
          });
          opts.appendChild(btn);
        });
        row.appendChild(opts);
        body.insertBefore(row, body.firstChild);
      });
    }

    root.querySelectorAll(".picker-opt").forEach(function (el) {
      var btn = /** @type {HTMLButtonElement} */ (el);
      var axis = btn.dataset.axis || "";
      var val = btn.dataset.value || "";
      var current = /** @type {Record<string, string>} */ (state)[axis];
      btn.classList.toggle("active", current === val);
      var disabled = (axis === "accel" && !combosValid(state.os, val));
      btn.disabled = disabled;
      btn.setAttribute("aria-disabled", disabled ? "true" : "false");
    });

    var cmd = generate();
    var out = root.querySelector("[data-picker-cmd]");
    if (out) out.innerHTML = colorize(cmd);
    var raw = /** @type {HTMLTextAreaElement | null} */ (root.querySelector("[data-picker-raw]"));
    if (raw) raw.value = cmd;

    var n = note();
    var nBox = /** @type {HTMLElement | null} */ (root.querySelector("[data-picker-note]"));
    if (nBox) {
      if (n) {
        nBox.hidden = false;
        nBox.className = "picker-note" + (n.kind === "warn" ? " warn" : "");
        nBox.innerHTML = n.text;
      } else {
        nBox.hidden = true;
      }
    }
    var preBox = root.querySelector("[data-picker-prereq]");
    if (preBox) {
      var preText = prereq();
      preBox.innerHTML = preText ? "<strong>Prereqs:</strong> " + preText : "";
    }
  }

  /**
   * Clipboard with fallback for insecure contexts (file://, old browsers).
   * @param {string} text
   * @returns {Promise<void>}
   */
  function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      try {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.top = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        // execCommand is deprecated but remains the only fallback for
        // non-secure contexts on desktop browsers.
        var ok = document.execCommand("copy");
        document.body.removeChild(ta);
        if (ok) resolve();
        else reject(new Error("execCommand copy failed"));
      } catch (e) {
        reject(e);
      }
    });
  }

  /**
   * @param {HTMLElement} btn
   * @param {string} label
   * @param {number} [ms]
   */
  function flashButton(btn, label, ms) {
    var original = btn.getAttribute("data-original-label");
    if (original === null) {
      original = btn.textContent || "";
      btn.setAttribute("data-original-label", original);
    }
    btn.classList.add("copied");
    btn.textContent = label;
    setTimeout(function () {
      btn.classList.remove("copied");
      btn.textContent = original;
    }, ms || 1500);
  }

  /**
   * @param {HTMLElement} root
   */
  function init(root) {
    render(root);

    var copyEl = /** @type {HTMLElement | null} */ (root.querySelector("[data-picker-copy]"));
    if (copyEl) {
      /** @type {HTMLElement} */
      var copyBtn = copyEl;
      copyBtn.addEventListener("click", function () {
        var raw = /** @type {HTMLTextAreaElement | null} */ (root.querySelector("[data-picker-raw]"));
        var text = raw ? raw.value : "";
        copyToClipboard(text)
          .then(function () { flashButton(copyBtn, "Copied!", 1600); })
          .catch(function (err) {
            console.error("Copy failed:", err);
            flashButton(copyBtn, "Copy failed", 1600);
          });
      });
    }
  }

  function initCopy() {
    document.querySelectorAll(".copy-btn[data-target]").forEach(function (el) {
      var btn = /** @type {HTMLElement} */ (el);
      btn.addEventListener("click", function () {
        var sel = btn.dataset.target;
        var target = sel ? document.querySelector(sel) : null;
        if (!target) return;
        var anchor = /** @type {HTMLElement} */ (target);
        var text = anchor.innerText || anchor.textContent || "";
        copyToClipboard(text)
          .then(function () { flashButton(btn, "Copied!"); })
          .catch(function (err) {
            console.error("Copy failed:", err);
            flashButton(btn, "Copy failed");
          });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var root = /** @type {HTMLElement | null} */ (document.querySelector("[data-picker]"));
    if (root) init(root);
    initCopy();
  });
})();
