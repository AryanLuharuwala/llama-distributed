package main

// Activation wire protocol (ACTV).
//
// Carries intermediate hidden states between pipeline stages and terminal
// logit/token frames back to the requester.  Rides the same WS binary channel
// as INFR frames; the 4-byte magic discriminates.  The framing is layout-
// compatible with a future WebRTC data-channel transport so we can swap the
// carrier without touching the producers/consumers.
//
// Frame layout (big-endian):
//
//   magic      u32   "ACTV"  (0x41435456)
//   ver        u8    0x01
//   type       u8    0x01 = activation,
//                    0x02 = token (terminal),
//                    0x03 = done,
//                    0x04 = error
//   req_id     u16
//   stage_idx  u16   producing stage (for activation/done/error) or 0xFFFF
//                    for a terminal token
//   tok_seq    u32   which token in the stream this belongs to
//   dtype      u8    0=f32, 1=f16, 2=bf16, 3=q8_0 (reserved), 4=bytes/utf8
//   rank       u8    number of tensor dimensions (0 for scalars)
//   flags      u8    bit0=is_prompt, bit1=kv_append, bit2=end_of_prompt
//   _rsvd      u8    0
//   dims[rank] u32   shape, each dim
//   payload_len u32
//   payload    bytes
//
// Header fixed portion is 20 bytes; dims add 4*rank; payload follows.

import (
	"encoding/binary"
	"errors"
)

const (
	actvMagic      = uint32(0x41435456) // "ACTV"
	actvVer        = uint8(0x01)
	actvTypeAct    = uint8(0x01)
	actvTypeToken  = uint8(0x02)
	actvTypeDone   = uint8(0x03)
	actvTypeError  = uint8(0x04)

	actvDTypeF32   = uint8(0)
	actvDTypeF16   = uint8(1)
	actvDTypeBF16  = uint8(2)
	actvDTypeQ8_0  = uint8(3)
	actvDTypeBytes = uint8(4)

	actvFlagIsPrompt     = uint8(0x01)
	actvFlagKVAppend     = uint8(0x02)
	actvFlagEndOfPrompt  = uint8(0x04)
)

// ActvFrame is the decoded/encodeable form.  Payload is owned by the caller;
// decode returns a slice aliasing the input buffer.
type ActvFrame struct {
	Type     uint8
	ReqID    uint16
	Stage    uint16
	TokSeq   uint32
	DType    uint8
	Flags    uint8
	Dims     []uint32
	Payload  []byte
}

// Encode writes the frame to a freshly-allocated byte slice.
// Layout: 20-byte fixed header + 4*rank dim bytes + 4-byte payload_len + payload.
func (f *ActvFrame) Encode() []byte {
	fixedHdr := 20
	dimsSize := 4 * len(f.Dims)
	totalHdr := fixedHdr + dimsSize + 4 // +4 for payload_len
	out := make([]byte, totalHdr+len(f.Payload))
	binary.BigEndian.PutUint32(out[0:], actvMagic)
	out[4] = actvVer
	out[5] = f.Type
	binary.BigEndian.PutUint16(out[6:], f.ReqID)
	binary.BigEndian.PutUint16(out[8:], f.Stage)
	binary.BigEndian.PutUint32(out[10:], f.TokSeq)
	out[14] = f.DType
	out[15] = uint8(len(f.Dims))
	out[16] = f.Flags
	out[17] = 0
	// Dims at offset 20.
	for i, d := range f.Dims {
		binary.BigEndian.PutUint32(out[20+4*i:], d)
	}
	// payload_len right after dims, at offset 20 + 4*rank.
	binary.BigEndian.PutUint32(out[fixedHdr+dimsSize:], uint32(len(f.Payload)))
	copy(out[totalHdr:], f.Payload)
	return out
}

// DecodeActvFrame parses a received frame.  Returns (nil, nil) if the magic
// doesn't match — the caller should treat it as "not mine" and fall through
// to another protocol handler.
func DecodeActvFrame(buf []byte) (*ActvFrame, error) {
	if len(buf) < 20 {
		return nil, nil
	}
	if binary.BigEndian.Uint32(buf[0:]) != actvMagic {
		return nil, nil
	}
	if buf[4] != actvVer {
		return nil, errors.New("actv: bad version")
	}
	f := &ActvFrame{
		Type:   buf[5],
		ReqID:  binary.BigEndian.Uint16(buf[6:]),
		Stage:  binary.BigEndian.Uint16(buf[8:]),
		TokSeq: binary.BigEndian.Uint32(buf[10:]),
		DType:  buf[14],
		Flags:  buf[16],
	}
	rank := int(buf[15])
	need := 20 + 4*rank + 4
	if len(buf) < need {
		return nil, errors.New("actv: truncated header")
	}
	if rank > 0 {
		f.Dims = make([]uint32, rank)
		for i := 0; i < rank; i++ {
			f.Dims[i] = binary.BigEndian.Uint32(buf[20+4*i:])
		}
	}
	payloadLen := int(binary.BigEndian.Uint32(buf[20+4*rank:]))
	hdr := 20 + 4*rank + 4
	if len(buf) < hdr+payloadLen {
		return nil, errors.New("actv: truncated payload")
	}
	f.Payload = buf[hdr : hdr+payloadLen]
	return f, nil
}

// ─── convenience constructors ──────────────────────────────────────────────

func NewActivationFrame(reqID uint16, stage uint16, tokSeq uint32,
	dtype uint8, dims []uint32, payload []byte, flags uint8) *ActvFrame {
	return &ActvFrame{
		Type:    actvTypeAct,
		ReqID:   reqID,
		Stage:   stage,
		TokSeq:  tokSeq,
		DType:   dtype,
		Flags:   flags,
		Dims:    dims,
		Payload: payload,
	}
}

func NewTokenFrame(reqID uint16, tokSeq uint32, text string) *ActvFrame {
	return &ActvFrame{
		Type:    actvTypeToken,
		ReqID:   reqID,
		Stage:   0xFFFF,
		TokSeq:  tokSeq,
		DType:   actvDTypeBytes,
		Payload: []byte(text),
	}
}

func NewDoneFrame(reqID uint16, tokSeq uint32) *ActvFrame {
	return &ActvFrame{
		Type:   actvTypeDone,
		ReqID:  reqID,
		Stage:  0xFFFF,
		TokSeq: tokSeq,
	}
}

func NewErrorFrame(reqID uint16, stage uint16, msg string) *ActvFrame {
	return &ActvFrame{
		Type:    actvTypeError,
		ReqID:   reqID,
		Stage:   stage,
		DType:   actvDTypeBytes,
		Payload: []byte(msg),
	}
}
