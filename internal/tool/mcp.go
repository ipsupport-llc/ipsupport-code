package tool

import (
	"context"
	"encoding/json"
)

// MCPListFunc returns a human-readable catalog of the configured MCP servers and
// their tools (the host owns the connections). MCPCallFunc invokes one tool;
// MCPSchemaFunc returns a tool's input schema on demand. The host gates calls
// with approval.
type (
	MCPListFunc   func(ctx context.Context) string
	MCPCallFunc   func(ctx context.Context, server, tool string, args map[string]any) (string, error)
	MCPSchemaFunc func(ctx context.Context, server, tool string) string
)

// NewMCP is the `mcp` proxy fat tool: ONE tool that fronts every configured MCP
// server, so the prompt catalog doesn't balloon with each server's schemas. The
// model calls list to discover tools, schema to see a tool's inputs, and call to
// run one. Only present when at least one MCP server is configured.
func NewMCP(list MCPListFunc, call MCPCallFunc, schema MCPSchemaFunc) Tool {
	return NewDomain(DomainSpec{
		Name:    "mcp",
		Summary: "Use tools from configured MCP servers: list them, see a tool's schema, or call one.",
		NotHere: "NOT here — local files → file; shell → run; web → web.",
		Actions: []Action{
			{Name: "list", Note: "(all servers + their tools — start here)", Run: func(ctx context.Context, _ Args) Result {
				return Ok(list(ctx))
			}},
			{Name: "schema", Params: []Param{Req("server", "str"), Req("tool", "str")}, Note: "(a tool's input schema)", Run: func(ctx context.Context, a Args) Result {
				return Ok(schema(ctx, a.Str("server"), a.Str("tool")))
			}},
			{Name: "call", Mutates: true, Params: []Param{Req("server", "str"), Req("tool", "str"), Opt("args", "str", "")}, Note: "(args = JSON object of the tool's inputs)", Run: func(ctx context.Context, a Args) Result {
				var args map[string]any
				if raw := a.Str("args"); raw != "" {
					if err := json.Unmarshal([]byte(raw), &args); err != nil {
						return Err("mcp call: 'args' must be a JSON object: " + err.Error())
					}
				}
				out, err := call(ctx, a.Str("server"), a.Str("tool"), args)
				if err != nil {
					return Err("mcp: " + err.Error())
				}
				return Ok(out)
			}},
		},
	})
}
