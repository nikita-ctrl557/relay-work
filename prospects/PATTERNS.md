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
