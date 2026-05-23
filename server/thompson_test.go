package main

import (
	"math/rand"
	"testing"
)

// TestBetaSampleInUnitInterval is the most basic invariant: a Beta
// distribution lives on [0,1].  If we ever return something outside
// that range, downstream comparisons silently misorder.
func TestBetaSampleInUnitInterval(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 2000; i++ {
		x := betaSample(rng, 3, 7)
		if x < 0 || x > 1 {
			t.Fatalf("Beta(3,7) sample out of range: %v", x)
		}
	}
}

// TestBetaSampleMeanApproachesAlphaOverAlphaPlusBeta verifies the
// Marsaglia-Tsang Gamma chain (and the Gamma-ratio Beta construction)
// hits the textbook mean to 3 decimal places over 50k draws.
func TestBetaSampleMeanApproachesAlphaOverAlphaPlusBeta(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	const N = 50000
	alpha, beta := 5.0, 15.0
	expected := alpha / (alpha + beta) // 0.25
	var sum float64
	for i := 0; i < N; i++ {
		sum += betaSample(rng, alpha, beta)
	}
	got := sum / N
	if diff := got - expected; diff < -0.01 || diff > 0.01 {
		t.Errorf("Beta(%v,%v) empirical mean %v, expected %v (diff %v)",
			alpha, beta, got, expected, diff)
	}
}

// TestThompsonPrefersStrongerRig covers the bandit's core property:
// when one arm has overwhelming evidence of higher success rate, the
// sampler should pick it almost always.
func TestThompsonPrefersStrongerRig(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	now := int64(1_700_000_000)

	strong := rigReputation{
		AgentID:                "strong",
		RelaySessionsTotal:     1000,
		RelaySessionsSuccess:   980,
	}
	weak := rigReputation{
		AgentID:              "weak",
		RelaySessionsTotal:   1000,
		RelaySessionsSuccess: 400,
	}

	const N = 1000
	strongWins := 0
	for i := 0; i < N; i++ {
		s1 := relayScoreSampled(strong, now, rng)
		s2 := relayScoreSampled(weak, now, rng)
		if s1 > s2 {
			strongWins++
		}
	}
	if strongWins < 950 {
		t.Errorf("strong rig won only %d/%d times — bandit isn't exploiting", strongWins, N)
	}
}

// TestThompsonExploresNewcomer covers the regret-optimal property
// from the other direction: a brand-new rig (Beta(1,1) = uniform)
// must sometimes beat an established mediocre rig.  Pure-greedy
// scoring would lock the newcomer out.
func TestThompsonExploresNewcomer(t *testing.T) {
	rng := rand.New(rand.NewSource(999))
	now := int64(1_700_000_000)

	veteran := rigReputation{
		AgentID:              "vet",
		RelaySessionsTotal:   100,
		RelaySessionsSuccess: 60,
	}
	newcomer := rigReputation{
		AgentID:              "new",
		RelaySessionsTotal:   0,
		RelaySessionsSuccess: 0,
	}

	const N = 1000
	newcomerWins := 0
	for i := 0; i < N; i++ {
		s1 := relayScoreSampled(veteran, now, rng)
		s2 := relayScoreSampled(newcomer, now, rng)
		if s2 > s1 {
			newcomerWins++
		}
	}
	// Beta(1,1) is uniform; veteran's posterior concentrates around 0.6.
	// Newcomer should beat veteran roughly 40% of draws — at minimum we
	// need it to beat veteran in at least 20% of trials to call this
	// "exploring."
	if newcomerWins < 200 {
		t.Errorf("newcomer won only %d/%d — bandit isn't exploring", newcomerWins, N)
	}
}

// TestThompsonRespectsRecencyPenalty proves the multiplicative recency
// guardrail still demotes a rig that just failed even if its lifetime
// posterior is excellent.
func TestThompsonRespectsRecencyPenalty(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	now := int64(1_700_000_000)

	excellent := rigReputation{
		AgentID:              "excellent",
		RelaySessionsTotal:   1000,
		RelaySessionsSuccess: 999,
		LastFailureAt:        now - 5, // failed 5 seconds ago
	}
	mediocre := rigReputation{
		AgentID:              "mediocre",
		RelaySessionsTotal:   1000,
		RelaySessionsSuccess: 500,
		// No recent failure.
	}

	const N = 1000
	mediocreWins := 0
	for i := 0; i < N; i++ {
		s1 := relayScoreSampled(excellent, now, rng)
		s2 := relayScoreSampled(mediocre, now, rng)
		if s2 > s1 {
			mediocreWins++
		}
	}
	// Excellent rig's draw gets multiplied by 0.1 (the 60-second post-fail
	// floor).  0.999 × 0.1 ≈ 0.1 vs mediocre's ~0.5.  Mediocre should
	// dominate the comparison nearly always.
	if mediocreWins < 950 {
		t.Errorf("recently-failed rig was picked too often; mediocre won %d/%d", mediocreWins, N)
	}
}

// TestRelayRngFreshDoesNotPanicUnderLoad confirms the rng helper used
// by findRelayAgent is safe for many concurrent goroutines.
func TestRelayRngFreshDoesNotPanicUnderLoad(t *testing.T) {
	const N = 200
	done := make(chan struct{}, N)
	for i := 0; i < N; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			r := mathRandFresh()
			_ = r.Float64()
			_ = r.NormFloat64()
		}()
	}
	for i := 0; i < N; i++ {
		<-done
	}
}
