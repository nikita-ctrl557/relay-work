# Prospecting patterns log

Read this before each run. Append findings — niches/geos/search patterns that
yield broken funnels, and operational blockers that waste a run.

## 2026-07-28 — Run 1: blocked before qualification, zero prospects added

**Environment cannot reach any of the mandated primary verification sources.**
This is the load-bearing finding from this run — record it so the next run
doesn't burn its whole budget rediscovering it.

Tried and confirmed blocked, from this cloud session (no local machine, no
browser tool, only WebFetch/WebSearch):

- `facebook.com/ads/library/*` (any query) → HTTP 403, no body.
- `facebook.com/<page>` and `m.facebook.com/<page>` → redirects to login wall
  (desktop) or "Facebook is not available on this browser" error (mobile).
  No page content, ever, logged out.
- `instagram.com/<handle>` and `instagram.com/explore/tags/*` → HTTP 429 on
  every distinct URL tried (not a one-off rate limit — persistent across
  ~4 different profiles/tags over the session).
- `adstransparency.google.com` → loads shell only, no advertiser data (JS-gated).
- Third-party IG viewers (picuki.com) → HTTP 403.
- `web.archive.org` → refused at the tool level, not fetchable at all.
- `bing.com/search` and `duckduckgo.com/html` via WebFetch → both serve a bot
  CAPTCHA challenge page instead of results.
- `WebSearch` (the search tool, not WebFetch) still returns snippets that
  *describe* Instagram/Facebook/Linktree pages (bio text, follower counts),
  but these are exactly the "search snippets and directories reproduce
  guessed data" the brief warns against — unverifiable without a re-fetch of
  the primary source, and that re-fetch is what's blocked.

Net effect: qualifier #1 (ad evidence) and #2 (funnel dead-end) both require
either the Meta Ad Library link caption or the Instagram/Facebook bio link —
both unreachable. Static, non-Meta pages (e.g. linktr.ee) fetch fine and can
be checked, but a live Linktree alone doesn't establish "buying traffic
badly" or "the ad points here" without the ad/profile it's linked from.

Two candidates were sanity-checked this way and both turned out disqualified
anyway once their Linktree was actually read (not fabricated, genuinely
checked): "Phoenix Rising Exteriors" is Stark County, OH despite the name,
and "Farence Fencing" is Jacksonville, FL — both have working marketing
sites with contact/quote pages, not dead ends.

**Action for next run:** before spending budget on sourcing, re-test whether
`facebook.com`/`instagram.com`/ad-library endpoints are reachable yet (proxy
or environment may change between sessions). If still blocked, this run's
approach (WebSearch → WebFetch verify) cannot produce compliant leads at
all — escalate to the user for an alternate path (e.g. a session with a real
browser tool, or manual spot-checks) rather than quietly relying on
unverified search snippets.

No prospects were added this run. Zero committed to inbox.json — nothing to
commit, per the brief's instruction not to pad.

## 2026-07-28 — Run 2: same blocker confirmed, plus additional channels tried

Re-tested the block from a fresh session per run 1's own instruction, and
tried several channels run 1 didn't reach for. All blocked identically:

- `facebook.com/ads/library/*` → still HTTP 403.
- `facebook.com/<page>` → still redirects to login wall.
- `mbasic.facebook.com/<page>` (new this run) → "Facebook is not available
  on this browser" error page, no content.
- `instagram.com/<handle>` (profile pages) → still HTTP 429, consistent
  across multiple distinct handles.
- `instagram.com/explore/tags/<tag>` → *did* return content this run (a
  WebFetch-summarized description of the hashtag feed), unlike run 1's 429
  on the same kind of URL. But a hashtag feed only yields a prose summary of
  "companies featured" with no verifiable handle, bio link, or follower
  count attached to any single business — not enough to qualify or reject a
  specific lead against qualifiers 1/2/4. Not usable for verification.
- IG mirror sites, new this run: `imginn.com/<handle>` → 403.
  `picnob.com/profile/<handle>` → 403. Same outcome as run 1's picuki.com.
- `adstransparency.google.com` → still shell-only, no advertiser data.
- New this run: live Google SERP for a trade query, checking for
  "Sponsored"/"Ad"-labeled results directly (an alternate to the Meta Ad
  Library, since Google Search Ads render inline in the SERP itself) →
  `google.com/search` returns an error/empty shell via WebFetch, no ads or
  organic results visible.
- New this run: `duckduckgo.com/html` → 302 redirect to
  `html.duckduckgo.com/html`, which then returns HTTP 503. Confirms run 1's
  "CAPTCHA-gated" finding under a different failure mode.
- `web.archive.org` → still refused at the tool level.

**Conclusion: this is not a transient rate limit or a one-off proxy hiccup.**
Two independent runs, on different days, testing seven+ distinct primary
verification surfaces (Meta Ad Library, Facebook pages incl. mobile-lite,
Instagram profiles, three IG mirror sites, Google Ads Transparency Center,
live Google/DuckDuckGo SERPs, Wayback Machine) all fail the same way both
times. The one partial exception — IG hashtag/explore pages sometimes
returning a prose summary — does not attach verifiable data to a specific
business and cannot satisfy the brief's per-lead verification bar.

Every qualifier in the brief routes through one of these blocked surfaces:
qualifier 1 (ad evidence) and qualifier 2 (funnel dead-end/link caption)
need the Meta Ad Library or the live IG/FB profile; nothing else discovered
so far substitutes for either. A business's own website (reachable) can
show a broken *organic* funnel, but not that paid traffic is landing there,
which is the actual thing Relay sells against.

**This blocker will not resolve by retrying the same approach a third time.**
Continuing to fire this routine on the current schedule/environment just
repeats this exact null result. Flagging to the user rather than running
again unattended — needs either (a) a session/environment with real browser
or authenticated-proxy access to Meta/Google ad surfaces, or (b) a revised
sourcing method that doesn't depend on those surfaces (e.g. a human forwards
candidate ad screenshots/links for this session to verify against the
business's own site, or a different tool is added to the session that can
reach these domains).

No prospects added this run either. Zero committed to inbox.json.

## 2026-07-29 — Run 5: found and diagnosed a real browser toolchain, but it's blocked by a specific, fixable proxy/TLS bug — not a generic wall

This run's environment description (system prompt) mentioned something the
first four runs never checked: a pre-installed Chromium at
`/opt/pw-browsers` plus a global Playwright npm package
(`playwright@1.56.1` under `/opt/node22/lib/node_modules`), reachable via
Bash. That's a *real browser*, which is exactly what runs 1-4 said was
missing and needed to solve qualifier 1/2 verification (Meta Ad Library
JS challenge, IG client-side rendering). Spent this run's budget
determining whether it actually closes the gap. Short answer: **not yet —
it's blocked by a specific TLS handshake bug between Chromium and this
session's egress proxy, not by policy.** Full diagnosis below so run 6
doesn't have to redo it.

**What was newly confirmed working (raw curl, not WebFetch tool):**
- Direct `curl` (not the WebFetch tool) with a real browser User-Agent
  reaches `www.instagram.com/<handle>/` and sometimes gets a genuine
  HTTP 200 instead of the 302-to-login-wall the WebFetch tool reported in
  prior runs. Tested 5 consecutive fetches of a real profile
  (`instagram.com/natgeo/`) — all 5 returned 200, ~600KB.
  **But this is a dead end for data extraction**: the HTML is Instagram's
  React app shell only — no `og:description`, no `biography`, no
  `external_url`, no follower count anywhere in the markup. Instagram
  serves profile data via client-side GraphQL/JS after load, not in the
  initial document, for logged-out non-JS requests. A 200 status here is
  not usable evidence of anything.
- Instagram's internal `web_profile_info` API
  (`instagram.com/api/v1/users/web_profile_info/?username=X` with an
  `x-ig-app-id` header — a technique used by many scraping libraries) is
  reachable through the proxy but returns a *server-side* error
  (`"Asset asset://laser.provider/ig_business_category_subvertical has
  been deleted. You cannot use this schema"`) on every known public app-id
  tried. This looks like Meta's own backend regression/deprecation, not an
  auth wall — not fixable from this end.
- `graph.facebook.com/.../instagram_oembed` and
  `instagram.com/api/v1/oembed/` need a valid post ID and (for the Graph
  API) a real access token — not usable without a specific target post and
  Meta app credentials Relay doesn't have.
- Facebook Ad Library's 403 is a **client-side JS bot challenge**, not a
  static block: the body is `<script>...fetch('/__rd_verify_...?
  challenge=3')...window.location.reload()</script>`. It needs a real JS
  engine to solve, confirming a browser really is the required tool here
  — curl/WebFetch structurally cannot do this regardless of headers.
- A raw `curl` fetch of `google.com/search?q=...` returns HTTP 200 but the
  body is obfuscated/minified JS (no static result links) — Google's SERP
  is not scrapable via plain curl either; the WebSearch tool remains the
  only usable channel for search, with its existing "unverifiable
  snippet" caveat from run 1.

**Why Chromium itself doesn't work here, precisely:** launched
`chromium-1194/chrome-linux/chrome` via Playwright with `--proxy-server=
http://127.0.0.1:43015` (this session's `$HTTPS_PROXY`) and
`--ignore-certificate-errors` (the proxy re-terminates TLS with its own CA
per `/root/.ccr/README.md`, which Chromium's own NSS store doesn't trust
by default — confirmed separately: a *direct*, non-proxied request to a
`no_proxy`-listed host like `pypi.org` still hit
`ERR_CERT_AUTHORITY_INVALID`, proving TLS interception is transparent at
the network level here, not just on the explicit proxy path). Tried ~10
flag combinations (disable QUIC/HTTP2, disable background networking/
component-update/domain-reliability/client-side-phishing, disable
post-quantum-Kyber key share, host-resolver-rules to kill Google's
background preconnects, `--no-sandbox`). Every single variant fails
identically: `net::ERR_CONNECTION_RESET`. Captured a full Chrome netlog
(`--log-net-log`) for the failing case: the HTTP CONNECT tunnel to the
proxy *succeeds* (confirmed independently with raw `nc` sending Chrome's
exact CONNECT headers, including its `HeadlessChrome` User-Agent — proxy
returns `200 Connection Established` every time, so it's not a UA-based
policy block). The reset happens *after* the tunnel is up, during the TLS
handshake inside it: `SSL_HANDSHAKE_ERROR net_error -101`,
`SOCKET_READ_ERROR os_error 104` (ECONNRESET), on every domain tried
(`example.com` included, as a neutral control — this is not
target-site-specific). Meanwhile `curl` completes real TLS handshakes
through the identical tunnel to the identical hosts without issue. The
working theory: Chromium's TLS 1.3 ClientHello is materially larger/more
complex than curl's (GREASE, ALPS, larger cipher/extension lists, and
until disabled, a post-quantum hybrid key share) and the proxy's
TLS-terminating egress hop resets the connection rather than parsing it —
a known class of bug in simpler MITM proxies that don't handle large or
fragmented ClientHellos. This is a proxy-side fix, not something more
Chrome flags can route around — confirmed by testing with post-quantum
Kyber explicitly disabled (`--disable-features=PostQuantumKyber,
UseMLKEM`) and it made no difference.

**Bottom line for run 6 and beyond:** don't re-attempt the Playwright
route with new flag combinations — the failure is proxy-side and
independent of Chrome's launch config, at least for every angle tried
this run. Don't bother with the IG "post URL sometimes loads" trick from
run 4 either — even a 200 status returns no embedded profile data, so it
can't establish bio link / follower count / DM-only CTA either. The
actual unblock, if one exists, is either: (a) whoever operates this
session's egress proxy fixes it to handle a full-size browser ClientHello
(this is a narrow, well-defined bug report — "TLS handshake reset for
large ClientHellos through the CONNECT tunnel," reproducible with the nc/
curl/Chrome comparison above), or (b) a future session gets a genuinely
different network path (e.g., a non-MITM'd egress, or a session type with
a browser tool built into the harness rather than raw Playwright over
this proxy).

Notifying the user by push: this is the first run with a *specific,
narrow, reproducible* technical root cause (proxy TLS-reset on large
ClientHello) rather than a vague "everything's blocked" — worth flagging
because it's fixable and different in kind from runs 1-4's escalations.

No prospects added this run. Zero committed to inbox.json — five
consecutive runs now with zero verifiable leads, all for the same
underlying reason.

## 2026-07-29 — Run 4: found a partial crack in the fetch wall, but it exposes a deeper sourcing problem, not a fix

Same day as run 3. Re-tested the core blockers first (Meta Ad Library `facebook.com/ads/library/*` → still 403; `instagram.com/<handle>` profile pages → still 429; `instagram.com/explore/tags/*` → still 429). No change there.

**New finding: individual Instagram post/reel URLs sometimes fetch successfully via WebFetch even though the profile page and hashtag page for the same account 429.** Tried `instagram.com/<handle>/reel/<id>/` and `instagram.com/p/<id>/` for ~8 distinct accounts found via WebSearch. Roughly half returned real content (caption, business name, sometimes phone/location) instead of a 429/403/empty-shell. The other half still failed (429, or a fetch that resolved to raw base64 image data with no text). Not reliable enough to depend on, but worth trying per-post before writing an account off — it's a genuine capability the last three runs didn't have documented.

**However, this doesn't solve the actual sourcing problem, and this run's real finding is why.** Used the post-URL trick plus targeted WebSearch queries (`"DM us for a quote"`, `"link in bio"`, `"message us"`, `linktr.ee <trade> <city>`, `"boosted post" <trade> <city>`) across pool fencing, roofing, fencing, outdoor kitchens, and hardscape in Phoenix metro, Texas (DFW, San Antonio), and Florida (Coral Springs). ~13 distinct candidate accounts surfaced and checked (WebSearch for a dedicated domain + BBB/Yelp listing, as a stand-in for the still-blocked live-profile check):

- Gr8 Glass Pool Fencing — real defect (DM-only CTA, no link) but Perth, Australia. Wrong geography, otherwise the best-shaped lead found all run.
- Wortham Brothers Roofing (TX), Glen C Landscape & Hardscape (Phoenix West Valley), Florida Outdoor Kitchens (Coral Springs), Texas Best Fence & Patio (DFW), Fence Pro's / Fence Pros of Texas (San Antonio), Boost Roofing (Tempe), Phoenix Roofing & Repair, Arizona Pool Fence — every one of these has a real, working, dedicated website with a contact/quote page (several also had BBB profiles and hundreds of Google reviews). All disqualify on qualifier 2 (no dead end) and most on qualifier 1 (sophisticated, not unsophisticated spend).

**Root cause, stated plainly: WebSearch-driven discovery is structurally biased against the exact profile the brief targets.** The qualifier is "Instagram/DM-only, no site, <10k followers, owner-run" — which is close to the definition of "not SEO'd, not indexed." A search engine surfaces what's indexed well; the businesses that rank for `"<trade> DM quote" <city>` are disproportionately the ones with a real marketing site *also* ranking nearby, not the ones with nothing but an Instagram profile. Every trade/geo combination tried this run reproduced this: the query returns 2-3 Instagram-only-looking results mixed with 5-8 fully-built competitor sites, and the Instagram-only-looking ones turn out, on the one check WebSearch permits (search for the business name + "website"), to have a site too — it just didn't rank on the first query.

**This means the blocker isn't just "can't fetch Meta/IG" anymore (run 4 shows partial workarounds exist for that) — it's "the discovery channel available (WebSearch) can't select for the target population in the first place."** A fix needs a discovery method that doesn't route through a general search engine's ranking: e.g. a real Meta Ad Library session (still 403), scrolling an Instagram hashtag/location feed as a human/browser would (still 429 on the feed endpoints, though individual posts sometimes load), or a seed list of candidate handles supplied by the user for this session to verify against.

No prospects added this run. Zero committed to inbox.json. Not re-notifying the user by push — run 3 already escalated this exact structural blocker today and the actionable ask (need a different discovery channel) hasn't changed; a second same-day ping would be noise. This entry exists so run 5 doesn't re-spend budget re-discovering the WebSearch bias from scratch and instead goes straight for a non-search-engine discovery method or asks for seed handles.

## 2026-07-29 — Run 3: blocker confirmed a third time, escalating to user via notification

This run had two working tools the first two runs lacked full clarity on: a
proper `WebSearch` tool (live, not just cached snippets) and `WebFetch`
against ordinary business domains (confirmed working — `example.com` fetched
cleanly). Neither closes the gap. Retested the core blocked surfaces plus a
few new ones:

- `facebook.com/ads/library/*` → still HTTP 403.
- `facebook.com/<page>` and `/about` → loads but truncated/no usable data;
  `mbasic.facebook.com/<page>` → still "Facebook is not available on this
  browser."
- `instagram.com/<handle>/embed/` (new) → "profile may be broken or removed"
  — an embed-specific gate, not evidence the account is actually gone.
- `threads.com/@<handle>` (new, redirects from threads.net) → login wall
  only, no bio/follower/link data.
- `instagram.com/explore/tags/*` → HTTP 429 this run (was intermittently
  readable in run 2; not reliable either way).
- `bing.com/search`, `duckduckgo.com` → still return CAPTCHA or irrelevant
  generic results, not real SERPs.
- `adstransparency.google.com` (new: queried by advertiser domain, not just
  the bare shell) → still no advertiser data, sign-in prompt only.
- `search.marginalia.nu` (new, indie search engine, tried on the theory that
  a non-commercial index might surface low-SEO small businesses better) —
  redirects fine but not pursued further once the FB/IG wall made the result
  moot either way for qualifiers 1/2.

New structural finding this run: even where `WebSearch`/`WebFetch` DO work
(ordinary sites, general queries), results systematically skew toward
businesses with real websites and decent SEO — e.g. a `"boosted post" OR
"link in bio"` query for Phoenix roofing/fencing/pool returned established
companies with full marketing sites, not the Instagram-only micro-businesses
the brief targets. This is not a fluke: the qualifier profile (no
site, Instagram/DM only, <10k followers) is close to invisible to
search-engine indexing by construction, so a search-engine-first sourcing
method fights its own target even before the Meta wall is considered.

**Conclusion:** three consecutive runs, three different days, now confirm
the same wall around the two mandatory verification surfaces (Meta Ad
Library for qualifier 1, live IG/FB profile or ad link caption for qualifier
2). This is not transient. Per run 2's own conclusion, continuing to fire
this routine unattended on the same approach cannot produce a compliant
lead — it can only burn budget re-confirming the same block. This run
notifies the user directly (push notification) rather than only logging to
a commit message nobody may read, since two prior silent log-only
escalations did not change the environment or the routine's schedule.

No prospects added this run. Zero committed to inbox.json.
