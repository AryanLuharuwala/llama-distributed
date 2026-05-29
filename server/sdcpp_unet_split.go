package main

// sdcpp_unet_split.go — CF12-W7 production coordinator for N-way distributed
// UNet generation (the "remote-denoise" path).
//
// One HOST rig runs sd.cpp's full sample() loop (TE + denoiser + sampler +
// VAE locally) but delegates every per-step UNet eval to this coordinator;
// N "unet_blocks" rigs each run a contiguous slice of the UNet. Per eval:
//
//   host --sdcpp_need_denoise{x,t,ctx,y,block_total}--> coordinator
//   coordinator: cond SDCD (ctx,y) + step_x SDCD (x) -> run the N-stage block
//                chain (chaining the carry frame) -> noise_pred (eps)
//   coordinator --sdr_denoise_result{eps}--> host  (mid-sample())
//
// The host emits sdcpp_done{png} when the full denoise + VAE finish. This is a
// direct port of python/dpp_runtime/validate_remote_denoise.py, which proved
// the mechanism end-to-end on rtxserver.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"strconv"
)

// ── minimal SDCD writer ──────────────────────────────────────────────────
// We only ever WRAP tensors whose bytes are already SDT frames (produced by
// the worker and shipped to us in the need_denoise frame), so no SDT parsing
// is needed — just the SDCD container envelope. Layout mirrors
// python/dpp_runtime/sdt_codec.py::sdcd_encode (all multi-byte fields BE).
const sdcdMagic uint32 = 0x53444344 // "SDCD"

type sdcdTensor struct {
	name string
	sdt  []byte // raw SDT1 frame bytes
}

func sdcdWrap(kv [][2]string, tensors []sdcdTensor) []byte {
	var b bytes.Buffer
	var u32 [4]byte
	var u16 [2]byte
	put32 := func(v uint32) { binary.BigEndian.PutUint32(u32[:], v); b.Write(u32[:]) }
	put16 := func(v uint16) { binary.BigEndian.PutUint16(u16[:], v); b.Write(u16[:]) }

	put32(sdcdMagic)
	b.WriteByte(1) // ver
	b.WriteByte(0) // flags
	put16(uint16(len(kv)))
	put16(uint16(len(tensors)))
	put16(0) // reserved
	for _, p := range kv {
		put16(uint16(len(p[0])))
		put16(uint16(len(p[1])))
		b.WriteString(p[0])
		b.WriteString(p[1])
	}
	for _, t := range tensors {
		put16(uint16(len(t.name)))
		b.WriteString(t.name)
		b.Write(t.sdt)
	}
	return b.Bytes()
}

// partitionBlocks splits [0,total) into `stages` contiguous ranges, front-
// loading the remainder (matches partitionUNetBlocks / the worker schedule).
func partitionBlocks(total, stages int) [][2]int {
	if total < stages || stages < 1 {
		return nil
	}
	out := make([][2]int, 0, stages)
	base, extra, cursor := total/stages, total%stages, 0
	for i := 0; i < stages; i++ {
		size := base
		if i < extra {
			size++
		}
		out = append(out, [2]int{cursor, cursor + size})
		cursor += size
	}
	return out
}

// dispatchSdcppWorkerCmd ships a raw worker stdin line to a rig, wrapped as a
// sdcpp_worker_cmd frame (gpunet-node's handle_sdcpp_worker_cmd writes cmd_line
// straight to the resident daemon).
func (s *server) dispatchSdcppWorkerCmd(target sdcppRoleAgent, reqID uint16, cmd map[string]any) bool {
	ac, ok := s.hub.findAgent(target.UserID, target.AgentID)
	if !ok {
		return false
	}
	line, err := json.Marshal(cmd)
	if err != nil {
		return false
	}
	ac.send(map[string]any{
		"kind":     "sdcpp_worker_cmd",
		"req_id":   int(reqID),
		"cmd_line": string(line),
	})
	return true
}

// dispatchSdcppBlockStage ships one stage of the block chain (role
// unet_blocks) — block_lo/block_hi out of block_total, with the cond SDCD and
// the per-step upld (step_x for stage 0, the prior stage's carry otherwise).
func (s *server) dispatchSdcppBlockStage(target sdcppRoleAgent, reqID uint16,
	modelPath, sdcdB64, upldB64 string, lo, hi, total int, t float64, sampler string) bool {
	ac, ok := s.hub.findAgent(target.UserID, target.AgentID)
	if !ok {
		return false
	}
	if sampler == "" {
		sampler = "euler_a"
	}
	ac.send(map[string]any{
		"kind":        "sdcpp_role_route",
		"req_id":      int(reqID),
		"role":        "unet_blocks",
		"model_path":  modelPath,
		"sdcd_b64":    sdcdB64,
		"upld_b64":    upldB64,
		"block_lo":    lo,
		"block_hi":    hi,
		"block_total": total,
		"steps":       1,
		"step_idx":    0,
		"timestep":    t,
		"sampler":     sampler,
		"cfg":         1.0,
		"seed":        0,
	})
	return true
}

// runSdcppUnetSplit drives a distributed-UNet generation: dispatch
// sdr_generate_remote to the host, service its per-step need_denoise requests
// by running the N-stage block chain, and return the final PNG.
func (s *server) runSdcppUnetSplit(ctx context.Context, reqID uint16,
	ch chan sdcppResultMsg, host sdcppRoleAgent, blockRigs []sdcppRoleAgent,
	body sdcppRequestBody) ([]byte, error) {

	stages := len(blockRigs)
	if stages < 2 {
		return nil, errors.New("unet split needs >= 2 block rigs")
	}
	modelPathOf := func(a sdcppRoleAgent) string {
		if a.ModelPath != "" {
			return a.ModelPath
		}
		return body.ModelPath
	}

	// Kick off the host's full generate; it will emit need_denoise per eval.
	if !s.dispatchSdcppWorkerCmd(host, reqID, map[string]any{
		"cmd":             "sdr_generate_remote",
		"req_id":          int(reqID),
		"model_path":      modelPathOf(host),
		"prompt":          body.Prompt,
		"negative_prompt": body.Negative,
		"width":           body.Width,
		"height":          body.Height,
		"steps":           body.Steps,
		"cfg":             body.CFG,
		"seed":            body.Seed,
		"sampler":         body.Sampler,
	}) {
		return nil, errors.New("sdcpp host rig offline at dispatch time")
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case m, ok := <-ch:
			if !ok {
				return nil, errors.New("sdcpp result channel closed")
			}
			switch m.Kind {
			case sdcppResultDone:
				if m.PNGB64 == "" {
					return nil, errors.New("sdcpp split: done with empty png")
				}
				return base64.StdEncoding.DecodeString(m.PNGB64)
			case sdcppResultError:
				return nil, errors.New("sdcpp split: " + m.ErrMsg)
			case sdcppResultNeedDenoise:
				eps, err := s.runSdcppBlockChainEval(ctx, reqID, ch, blockRigs, modelPathOf, m)
				if err != nil {
					return nil, err
				}
				if !s.dispatchSdcppWorkerCmd(host, reqID, map[string]any{
					"cmd":     "sdr_denoise_result",
					"req_id":  int(reqID),
					"eps_b64": eps,
				}) {
					return nil, errors.New("sdcpp host rig offline mid-denoise")
				}
			default:
				// progress — keep waiting
			}
		}
	}
}

// runSdcppBlockChainEval services one need_denoise: builds the cond + step_x
// SDCD frames and runs the N contiguous block stages, chaining the carry.
// Returns the final noise_pred (eps) as base64 SDT.
func (s *server) runSdcppBlockChainEval(ctx context.Context, reqID uint16,
	ch chan sdcppResultMsg, blockRigs []sdcppRoleAgent,
	modelPathOf func(sdcppRoleAgent) string, ev sdcppResultMsg) (string, error) {

	total := ev.BlockTotal
	stages := len(blockRigs)
	ranges := partitionBlocks(total, stages)
	if ranges == nil {
		return "", errors.New("sdcpp split: bad block_total " + strconv.Itoa(total) +
			" for " + strconv.Itoa(stages) + " stages")
	}

	ctxSDT, _ := base64.StdEncoding.DecodeString(ev.NeedCtxB64)
	xSDT, _ := base64.StdEncoding.DecodeString(ev.NeedXB64)
	condTensors := []sdcdTensor{{name: "cond.crossattn", sdt: ctxSDT}}
	if ev.NeedYB64 != "" {
		if ySDT, _ := base64.StdEncoding.DecodeString(ev.NeedYB64); len(ySDT) > 0 {
			condTensors = append(condTensors, sdcdTensor{name: "cond.vector", sdt: ySDT})
		}
	}
	sdcdB64 := base64.StdEncoding.EncodeToString(sdcdWrap([][2]string{{"role", "te"}}, condTensors))
	stepX := base64.StdEncoding.EncodeToString(sdcdWrap(
		[][2]string{
			{"kind", "sdcpp_step_x"},
			{"step_idx", "0"},
			{"timestep", strconv.FormatFloat(ev.NeedT, 'g', -1, 64)},
		},
		[]sdcdTensor{{name: "x", sdt: xSDT}}))

	upld := stepX
	for i, r := range ranges {
		if !s.dispatchSdcppBlockStage(blockRigs[i], reqID, modelPathOf(blockRigs[i]),
			sdcdB64, upld, r[0], r[1], total, ev.NeedT, "euler_a") {
			return "", errors.New("sdcpp block rig offline at stage " + strconv.Itoa(i))
		}
		frame, err := waitForRoleFrame(ctx, ch, "unet_blocks")
		if err != nil {
			return "", err
		}
		upld = frame // chain the carry; the final stage's frame is the eps SDT
	}
	return upld, nil
}
