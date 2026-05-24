# Shared OCI image macro — keeps every image rule consistent.
#
# Each Go binary gets the same packaging treatment:
#   1. go_binary with stamped version (-X main.gitSHA=...)
#   2. tar of the static binary (deterministic mtime)
#   3. oci_image on top of distroless-base, ENTRYPOINT = the binary
#   4. oci_image_index for multi-arch (linux/amd64, linux/arm64)
#   5. cosign_sign rule wrapping the image with keyless OIDC signing
#
# Why this lives in tools/ rather than a per-package BUILD: the same
# six-rule incantation appears for dist-server, dist-turn, surd, and
# every future Go binary.  A macro keeps the per-package BUILD short
# and prevents drift in the supply-chain settings.

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_image_index", "oci_push")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

def dist_go_image(
        name,
        binary,
        repository,
        base = "@distroless_base",
        extra_layers = None,
        ports = None,
        env = None,
        labels = None,
        entrypoint_args = None,
        visibility = None):
    """Package a Go binary as a signed, multi-arch OCI image.

    Targets emitted (with `name = "dist-server"`):
      - `dist-server_layer`        — tar layer with the binary
      - `dist-server_image_amd64`  — single-arch image
      - `dist-server_image_arm64`  — single-arch image
      - `dist-server` aka image    — multi-arch oci_image_index
      - `dist-server_image_signed` — image + cosign attestation
      - `dist-server_push`         — `bazel run :dist-server_push` to
        push to the registry; signature pushed separately via the
        deploy/sign/cosign-sign.sh script.

    Args:
      binary:      the go_binary label to package
      repository:  registry path, e.g. "ghcr.io/llama-distributed/dist-server"
      base:        OCI base image label (default: distroless_base)
      extra_layers: additional pkg_tar layers (configs, certs)
      ports:       list of EXPOSE ports as strings
      env:         dict of ENV keys → values
      labels:      OCI labels dict (org.opencontainers.image.*)
      entrypoint_args: extra args to bake into ENTRYPOINT
    """
    pkg_tar(
        name        = name + "_layer",
        srcs        = [binary],
        package_dir = "/usr/local/bin",
        mode        = "0755",
    )

    layers = [":" + name + "_layer"]
    if extra_layers:
        layers += extra_layers

    entrypoint = ["/usr/local/bin/" + name]
    if entrypoint_args:
        entrypoint += entrypoint_args

    for arch in ("amd64", "arm64"):
        oci_image(
            name       = name + "_image_" + arch,
            base       = base + "_linux_" + arch,
            entrypoint = entrypoint,
            tars       = layers,
            exposed_ports = ports,
            env           = env,
            labels        = labels,
            # SOURCE_DATE_EPOCH=0 (set in .bazelrc) makes layer mtimes
            # deterministic, which means image digests are stable
            # across rebuilds — a precondition for cosign to keep
            # working after a CI re-run.
        )

    oci_image_index(
        name   = name,
        images = [
            ":" + name + "_image_amd64",
            ":" + name + "_image_arm64",
        ],
        visibility = visibility,
    )

    # Cosign signing is wired through a genrule that calls the
    # deploy/sign/cosign-sign.sh wrapper, which in turn delegates to
    # the cosign binary using either keyless OIDC (CI) or a key file
    # (local).  See deploy/sign/README.md for the threat model.
    native.genrule(
        name    = name + "_image_signed",
        srcs    = [":" + name],
        outs    = [name + "_sig.json"],
        cmd     = """
            $(location //deploy/sign:cosign-sign) \\
                --image $(SRCS) \\
                --repository """ + repository + """ \\
                --output $@
        """,
        tools   = ["//deploy/sign:cosign-sign"],
        executable = False,
        visibility = visibility,
        tags = ["manual", "no-sandbox"],  # needs network to call OIDC
    )

    oci_push(
        name       = name + "_push",
        image      = ":" + name,
        repository = repository,
        visibility = visibility,
    )

    # Alias the index as `:image` so the root BUILD's all_images
    # filegroup can refer to `//pkg:image` without the binary's name.
    native.alias(
        name       = "image",
        actual     = ":" + name,
        visibility = visibility,
    )
    native.alias(
        name       = "image_signed",
        actual     = ":" + name + "_image_signed",
        visibility = visibility,
    )
