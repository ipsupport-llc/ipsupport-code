//go:build !windows && !darwin

package hostarch

func native() string { return "" }
