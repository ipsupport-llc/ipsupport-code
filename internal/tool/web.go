package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/PuerkitoBio/goquery"
	readability "github.com/go-shiori/go-readability"
)

// Endpoints are package vars so tests can point them at an httptest server.
var (
	ddgSearchURL = "https://html.duckduckgo.com/html/"
	stackExURL   = "https://api.stackexchange.com/2.3/search/advanced"
)

const (
	maxHTTPBytes  = 2 << 20 // 2 MiB read cap per response
	maxFetchBytes = 40_000  // Markdown cap returned to the model
	userAgent     = "ipsupport-code/0.1 (+https://github.com/ipsupport-llc/ipsupport-code)"
)

type webTool struct{ hc *http.Client }

// NewWeb returns the web tool. A nil client uses http.DefaultClient.
func NewWeb(hc *http.Client) Tool {
	if hc == nil {
		hc = http.DefaultClient
	}
	return &webTool{hc: hc}
}

func (*webTool) Name() string      { return "web" }
func (*webTool) Actions() []string { return []string{"search", "fetch", "stackexchange"} }

func (*webTool) Description() string {
	return strings.TrimSpace(`Reach the live web: search, read a page as Markdown, or query StackExchange Q&A.
Actions:
  - search:        {"query": str, "limit"?: int=8}   web search (DuckDuckGo)
  - fetch:         {"url": str}                       fetch a page → clean Markdown
  - stackexchange: {"query": str, "site"?: str="stackoverflow", "tag"?: str, "limit"?: int=5}
Use search to find pages, fetch to read one, stackexchange for programming Q&A.
NOT here — local files → file; shell → run; arithmetic → calc.`)
}

func (w *webTool) Call(ctx context.Context, action string, params map[string]any) Result {
	switch action {
	case "search":
		return w.search(ctx, params)
	case "fetch":
		return w.fetch(ctx, params)
	case "stackexchange":
		return w.stackexchange(ctx, params)
	}
	return Err("web: unknown action " + action)
}

func (w *webTool) search(ctx context.Context, params map[string]any) Result {
	if err := Require(params, "query"); err != nil {
		return Err(err.Error())
	}
	q := Str(params, "query")
	limit := Int(params, "limit", 8)
	if limit < 1 {
		limit = 8
	}

	u, _ := url.Parse(ddgSearchURL)
	qs := u.Query()
	qs.Set("q", q)
	u.RawQuery = qs.Encode()

	body, err := w.get(ctx, u.String())
	if err != nil {
		return Fail("web", "search", "search request failed: "+err.Error(), err)
	}
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(body))
	if err != nil {
		return Fail("web", "search", "could not parse results", err)
	}

	titles := doc.Find(".result__a")
	snippets := doc.Find(".result__snippet")
	var lines []string
	titles.EachWithBreak(func(i int, a *goquery.Selection) bool {
		title := strings.TrimSpace(a.Text())
		if title == "" {
			return true
		}
		href, _ := a.Attr("href")
		snippet := strings.TrimSpace(snippets.Eq(i).Text())
		lines = append(lines, fmt.Sprintf("%s — %s\n  %s", title, decodeDDG(href), snippet))
		return len(lines) < limit
	})
	if len(lines) == 0 {
		return Ok("no results for: " + q)
	}
	return Ok(strings.Join(lines, "\n"))
}

func (w *webTool) fetch(ctx context.Context, params map[string]any) Result {
	if err := Require(params, "url"); err != nil {
		return Err(err.Error())
	}
	raw := Str(params, "url")
	pu, err := url.Parse(raw)
	if err != nil || (pu.Scheme != "http" && pu.Scheme != "https") {
		return Err("invalid url (need http/https): " + raw)
	}

	body, err := w.get(ctx, raw)
	if err != nil {
		return Fail("web", "fetch", "fetch failed: "+err.Error(), err)
	}

	htmlContent, title := body, ""
	if art, rerr := readability.FromReader(strings.NewReader(body), pu); rerr == nil && strings.TrimSpace(art.Content) != "" {
		htmlContent, title = art.Content, art.Title
	}
	markdown, cerr := md.NewConverter("", true, nil).ConvertString(htmlContent)
	if cerr != nil {
		markdown = stripTags(body)
	}
	markdown = strings.TrimSpace(markdown)
	if len(markdown) > maxFetchBytes {
		markdown = markdown[:maxFetchBytes] + "\n…[truncated]"
	}
	if title != "" {
		return Ok("# " + title + "\n\n" + markdown)
	}
	return Ok(markdown)
}

func (w *webTool) stackexchange(ctx context.Context, params map[string]any) Result {
	if err := Require(params, "query"); err != nil {
		return Err(err.Error())
	}
	q := Str(params, "query")
	site := Str(params, "site")
	if site == "" {
		site = "stackoverflow"
	}
	limit := Int(params, "limit", 5)
	if limit < 1 {
		limit = 5
	}

	u, _ := url.Parse(stackExURL)
	qs := u.Query()
	qs.Set("order", "desc")
	qs.Set("sort", "relevance")
	qs.Set("q", q)
	qs.Set("site", site)
	if tag := Str(params, "tag"); tag != "" {
		qs.Set("tagged", tag)
	}
	qs.Set("pagesize", strconv.Itoa(limit))
	u.RawQuery = qs.Encode()

	body, err := w.get(ctx, u.String())
	if err != nil {
		return Fail("web", "stackexchange", "request failed: "+err.Error(), err)
	}
	var resp struct {
		Items []struct {
			Title       string `json:"title"`
			Link        string `json:"link"`
			Score       int    `json:"score"`
			AnswerCount int    `json:"answer_count"`
			IsAnswered  bool   `json:"is_answered"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return Fail("web", "stackexchange", "could not decode response", err)
	}
	if len(resp.Items) == 0 {
		return Ok("no StackExchange results for: " + q)
	}
	var lines []string
	for i, it := range resp.Items {
		if i >= limit {
			break
		}
		ans := "unanswered"
		if it.IsAnswered {
			ans = "answered"
		}
		lines = append(lines, fmt.Sprintf("%s — %s (score %d, %d answers, %s)",
			html.UnescapeString(it.Title), it.Link, it.Score, it.AnswerCount, ans))
	}
	return Ok(strings.Join(lines, "\n"))
}

func (w *webTool) get(ctx context.Context, urlStr string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := w.hc.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxHTTPBytes))
	if err != nil {
		return "", err
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("http %d", resp.StatusCode)
	}
	return string(data), nil
}

// decodeDDG unwraps DuckDuckGo's redirect href (…/l/?uddg=<encoded url>).
func decodeDDG(href string) string {
	if href == "" {
		return ""
	}
	if i := strings.Index(href, "uddg="); i >= 0 {
		raw := href[i+len("uddg="):]
		if amp := strings.IndexByte(raw, '&'); amp >= 0 {
			raw = raw[:amp]
		}
		if dec, err := url.QueryUnescape(raw); err == nil {
			return dec
		}
	}
	if strings.HasPrefix(href, "//") {
		return "https:" + href
	}
	return href
}

var tagRE = regexp.MustCompile(`<[^>]+>`)

func stripTags(s string) string { return strings.TrimSpace(tagRE.ReplaceAllString(s, "")) }
