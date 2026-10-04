//go:build !darwin && !linux && !windows

package telemetry

func platformInfo() (osVersion, chip string, memoryGB int) { return "", "", 0 }
