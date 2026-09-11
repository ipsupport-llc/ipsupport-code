//go:build !windows

package procgroup

import (
	"context"
	"os/exec"
	"testing"
	"time"
)

// Set's whole point: killing only the direct child leaves a grandchild that's
// still holding the output pipe running, so cmd.Wait() never returns until
// WaitDelay forcibly steps in — a task looks wedged for that whole span. This
// drives a real shell that backgrounds a long-lived grandchild and confirms
// cancellation reaps the WHOLE group promptly, well under WaitDelay.
func TestSetKillsWholeGroupOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, "sh", "-c", "sleep 30 & wait")
	Set(cmd)
	if _, err := cmd.StdoutPipe(); err != nil { // give the grandchild a pipe to hold open
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond) // let the backgrounded sleep actually start

	cancel()
	done := make(chan struct{})
	go func() { cmd.Wait(); close(done) }()

	select {
	case <-done:
		// Group killed promptly — the grandchild's pipe end closed with it.
	case <-time.After(3 * time.Second):
		t.Fatal("cmd.Wait() did not return within 3s of cancel — the grandchild is likely still " +
			"running and holding the pipe (Set isn't killing the whole process group)")
	}
}

// Set must configure both halves of its contract: a fresh process group (so
// Cancel below can target it) and a bounded Wait (so a stuck pipe holder that
// somehow survives the group kill can't hang the caller forever).
func TestSetConfiguresProcessGroupAndWaitDelay(t *testing.T) {
	cmd := exec.Command("true")
	Set(cmd)
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.Setpgid {
		t.Error("Set must set SysProcAttr.Setpgid — without it, cmd is not the leader of its own process group")
	}
	if cmd.Cancel == nil {
		t.Error("Set must set Cancel — without it, ctx cancellation falls back to killing only the direct child")
	}
	if cmd.WaitDelay != 5*time.Second {
		t.Errorf("WaitDelay = %v, want 5s", cmd.WaitDelay)
	}
}
