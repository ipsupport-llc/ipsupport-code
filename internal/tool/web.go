package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
	"github.com/PuerkitoBio/goquery"
	// go-shiori/go-readability is unmaintained upstream; kept for v1 since fetch
	// degrades gracefully (readability failure → raw HTML → stripTags). Future
	// swap candidate: codeberg.org/readeck/go-readability/v2.
	readability "github.com/go-shiori/go-readability"

	"github.com/ipsupport-llc/ipsupport-code/internal/textutil"
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
	w := &webTool{hc: hc}
	return NewDomain(DomainSpec{
		Name:    "web",
		Summary: "Reach the live web: search, read a page as Markdown, or query StackExchange Q&A.",
		Details: "Use search to find pages, fetch to read one, stackexchange for programming Q&A.",
		NotHere: "NOT here — local files → file; shell → run; arithmetic → calc.",
		Actions: []Action{
			{Name: "search", Params: []Param{Req("query", "str"), Opt("limit", "int", "8")}, Run: w.search},
			{Name: "fetch", Params: []Param{Req("url", "str")}, Run: w.fetch},
			{Name: "stackexchange", Params: []Param{Req("query", "str"), Opt("site", "str", "stackoverflow"), Opt("tag", "str", ""), Opt("limit", "int", "5")}, Run: w.stackexchange},
		},
	})
}

func (w *webTool) search(ctx context.Context, a Args) Result {
	q := a.Str("query")
	limit := a.Int("limit", 8)
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

func (w *webTool) fetch(ctx context.Context, a Args) Result {
	raw := a.Str("url")
	pu, err := url.Parse(raw)
	if err != nil || (pu.Scheme != "http" && pu.Scheme != "https") {
		return Err("invalid url (need http/https): " + raw)
	}

	if !webAllowPrivate && privateHost(pu.Host) {
		return Err("refusing to fetch a private/loopback/link-local address (SSRF guard): " + pu.Host)
	}

	body, err := w.fetchGet(ctx, raw)
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
	if clipped, truncated := textutil.Clip(markdown, maxFetchBytes); truncated {
		markdown = clipped + "\n…[truncated]"
	}
	if title != "" {
		return Ok("# " + title + "\n\n" + markdown)
	}
	return Ok(markdown)
}

func (w *webTool) stackexchange(ctx context.Context, a Args) Result {
	q := a.Str("query")
	site := a.Str("site")
	if site == "" {
		site = "stackoverflow"
	}
	limit := a.Int("limit", 5)
	if limit < 1 {
		limit = 5
	}

	u, _ := url.Parse(stackExURL)
	qs := u.Query()
	qs.Set("order", "desc")
	qs.Set("sort", "relevance")
	qs.Set("q", q)
	qs.Set("site", site)
	if tag := a.Str("tag"); tag != "" {
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

// fetchGet is get for model-supplied URLs: it re-checks every redirect hop so a
// public URL can't bounce to a private/loopback/link-local address (SSRF).
func (w *webTool) fetchGet(ctx context.Context, urlStr string) (string, error) {
	client := *w.hc // copy: don't mutate the shared client's CheckRedirect
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if !webAllowPrivate && privateHost(req.URL.Host) {
			return fmt.Errorf("redirect to a private address blocked: %s", req.URL.Host)
		}
		if len(via) >= 10 {
			return fmt.Errorf("stopped after 10 redirects")
		}
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
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

// webAllowPrivate disables the SSRF guard (tests point fetch at loopback servers).
var webAllowPrivate bool

// privateHost reports whether host resolves to a loopback/private/link-local/
// unspecified address — the SSRF targets fetch refuses (cloud metadata at
// 169.254.169.254 is link-local; internal services are loopback/private).
func privateHost(host string) bool {
	h := host
	if hh, _, err := net.SplitHostPort(host); err == nil {
		h = hh
	}
	if ip := net.ParseIP(h); ip != nil {
		return blockedIP(ip)
	}
	ips, err := net.LookupIP(h)
	if err != nil {
		return false // can't resolve → let the request fail normally
	}
	for _, ip := range ips {
		if blockedIP(ip) {
			return true
		}
	}
	return false
}

func blockedIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() ||
		ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast()
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
