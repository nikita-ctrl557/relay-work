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
