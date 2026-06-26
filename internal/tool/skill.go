package tool

import "context"

// SkillSource is the read-only view of the skill store the model is allowed to
// touch: see the catalog of enabled skills and load one's full instructions.
// Installing and toggling skills is a human action (the /skills command), so it
// is deliberately NOT reachable from here.
type SkillSource interface {
	Index() string
	Body(name string) (string, error)
}

// NewSkill exposes enabled skills to the model. It is only registered when at
// least one skill is enabled, so it costs nothing in the catalog otherwise.
func NewSkill(src SkillSource) Tool {
	return NewDomain(DomainSpec{
		Name:    "skill",
		Summary: "On-demand instruction packs. When a skill's topic fits the task, load it and follow it.",
		NotHere: "NOT here — the user installs/toggles skills via /skills; you only list and load.",
		Actions: []Action{
			{
				Name: "list",
				Note: "(enabled skills and what they're for)",
				Run: func(_ context.Context, _ Args) Result {
					if idx := src.Index(); idx != "" {
						return Ok(idx)
					}
					return Ok("(no skills enabled)")
				},
			},
			{
				Name:   "load",
				Params: []Param{Req("name", "str")},
				Note:   "(returns the skill's full instructions to follow now)",
				Run: func(_ context.Context, a Args) Result {
					body, err := src.Body(a.Str("name"))
					if err != nil {
						return Err(err.Error())
					}
					return Ok(body)
				},
			},
		},
	})
}
