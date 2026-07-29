# Prospecting patterns log

Read this before each run. Append findings — niches/geos/search patterns that
yield broken funnels, and operational blockers that waste a run.

## 2026-07-29 — Run 12: blocker unchanged, minimal spot-check only, no push (nothing new since run 10's escalation)

Ran the same minimal spot-check as run 11, since runs 1-11 already exhausted
the realistic channel list and reconfirmed it repeatedly:

- `facebook.com/ads/library/*` via raw curl → still HTTP 403. Unchanged.
- `instagram.com/natgeo/` via curl with mobile Safari UA → still HTTP 200,
  `og:description` follower count readable (269M Followers — matches real
  IG data, confirms run 8's crack is still live for qualifier 4 only), but
  no bio/link/business-contact field anywhere in the response, same as
  every run since 8. Still doesn't touch qualifiers 1/2.
- `adstransparency.google.com` → still HTTP 200 shell.
- `ListConnectors` filtered for meta/facebook/instagram/ads/browser → only
  Gmail connected. No ads-data connector exists on this account. Same as
  runs 7/11.
- Retried the Playwright/Chromium route from run 5 fresh (proxy fix check,
  per run 10's own recommended trigger) → this time it didn't even reach
  run 5/10's `ERR_CONNECTION_RESET` error message within a 30s timeout; the
  launch just hung and was killed. Consistent with "still broken," not a
  new failure mode — not worth deeper diagnosis, the underlying proxy bug
  was already root-caused in run 5 and confirmed unfixed in run 10.

No new channel, no new capability, no change in any of the three canonical
blockers. **No push notification sent** — run 10 already delivered the
actionable escalation (hourly cadence + proxy TLS-reset root cause) two
cycles ago, and runs 11-12 confirm nothing has changed that would give the
user new information to act on. Repeating that ping every run is exactly
the noise runs 6/7/9/11 already decided against.

No prospects added. Twelve consecutive runs, zero verifiable leads, same
root cause throughout: qualifiers 1 (ad spend evidence) and 2 (funnel
dead-end/link caption) both require either the Meta Ad Library or a live
IG/FB profile's bio/link, and nothing reachable from this environment can
render either. Not committing prospects/inbox.json (still doesn't exist).
Committing only this log entry.

**Unchanged standing recommendation:** this cron (hourly) will keep
producing an identical null result every run until one of: (a) the egress
proxy's TLS-reset on large ClientHellos gets fixed so a real browser can
clear Meta's JS challenge, (b) the schedule is dropped to something far
less frequent than hourly, or (c) a human seeds candidate handles/ad
screenshots for this session to verify against the business's own site
instead of sourcing from Ad Library/IG directly.

## 2026-07-29 — Run 11: blocker unchanged, minimal spot-check only (per own prior guidance not to re-burn budget re-diagnosing)

Did not repeat the full diagnostic sweep — runs 1-10 already exhausted the
realistic channel list (Ad Library direct, IG profile/API/legacy-AJAX, IG
mirrors, TikTok/Google Ads Transparency, Yelp/BBB/Maps, SERPs, Wayback,
Playwright/Chromium via the proxy) and run 10 reconfirmed all of them fresh
just one cycle ago. Instead ran the three cheapest possible spot-checks to
see if anything shifted in the last hour:

- `facebook.com/ads/library/*` via WebFetch AND raw curl → still HTTP 403
  on both. Unchanged.
- `instagram.com/natgeo/` via curl with mobile Safari UA → still HTTP 200
  shell-only (consistent with runs 5-10's finding that this status code
  carries no bio/link/ad data, only og:description follower counts).
- `ListConnectors` filtered for meta/facebook/instagram/ads → only Gmail
  is connected; no Meta/Instagram/ads-data connector exists on this
  account. Confirms run 7's same check, still true.

No new channel attempted, no new capability found. **Not sending a push
notification** — run 10 already delivered the actionable escalation
(hourly cadence + confirmed-unfixed proxy bug) one cycle ago, and nothing
has changed since that would give the user new information to act on.
Repeating that ping every hour with no new content is exactly the noise
runs 6/7/9 already decided against.

No prospects added. Eleven consecutive runs, zero verifiable leads, same
root cause. Not committing prospects/inbox.json (still doesn't exist —
nothing to add). Committing only this log entry.

**Standing recommendation for whoever reads this next, unchanged from run
10:** this cron (`13 * * * *`, hourly) will keep producing an identical
null result every run until one of: (a) the egress proxy's TLS-reset on
large ClientHellos gets fixed so a real browser can clear Meta's JS
challenge, (b) the schedule is dropped to something far less frequent than
hourly, or (c) a human seeds candidate handles/ad screenshots for this
session to verify against the business's own site instead of sourcing from
Ad Library/IG directly.

## 2026-07-29 — Run 10: blocker unchanged (10th straight run); this trigger fires HOURLY, not daily — flagged to user

Discovered via `list_triggers` this run: the cron behind this routine is
`13 * * * *` — **hourly**, created 2026-07-28T21:13, i.e. roughly one run
every hour since yesterday evening. That's the real cadence behind runs
1-10, not a daily job. Worth recording because it changes the cost/benefit
of continuing to fire unattended on the current approach: ten straight
hourly-ish runs have produced zero prospects for the same structural reason.

Re-tested the three canonical blockers fresh (not reusing old output):
- `facebook.com/ads/library/*` → still HTTP 403, byte-identical JS
  challenge (`fetch('/__rd_verify_...?challenge=3')`). Unchanged since run 1.
- `instagram.com/<handle>` via curl, mobile Safari UA → HTTP 302 to login
  (matches run 9, not run 8's brief 200+og:description crack — that crack
  has now failed to reproduce twice in a row across two different days).
- `adstransparency.google.com` → HTTP 200, 2.5MB JS shell, no advertiser data.
- Retested the Playwright/Chromium route from run 5 fresh, in case the
  proxy bug had been fixed since: identical failure, byte-for-byte —
  `net::ERR_CONNECTION_RESET` mid-TLS-handshake through the CONNECT tunnel,
  on a neutral control domain (`example.com`), using the exact repro from
  run 5. **Not fixed.**

Three genuinely new channels tried this run, all dead ends:
- `dumpor.io/v/<handle>` (IG mirror, not tried by runs 1-9) — actually
  loads (HTTP 200, real HTML, correct page title) after a redirect from
  `dumpor.com`, unlike every other IG mirror tried across all runs. But
  it's Cloudflare-Turnstile-gated for the actual profile data — no
  follower count, bio, or external link anywhere in the served HTML, only
  UI chrome. Not usable.
- `pixwox.com` → redirects to `picnob.com`, HTTP 403 (same class as run 6's
  picuki.com finding).
- `gramho.com` → proxy CONNECT tunnel itself fails (502), can't even reach
  the TLS layer.
- `web.archive.org` (direct curl, not via WebFetch tool) — this actually
  works (HTTP 404 on the specific query tried, real Wayback HTML, not
  "refused at the tool level" as run 1 reported for the WebFetch tool
  specifically). But moot for this brief: Ad Library search-result pages
  are dynamic/behind a JS challenge, so nothing meaningful would ever have
  been archived for a given query even if a snapshot existed.
- `site:facebook.com/ads/library <trade> <city>` and `"DM us for a quote"
  ... instagram` via WebSearch (not WebFetch) — reproduced the *exact same
  two* Phoenix pool-fence Instagram accounts (phoenixpoolfence,
  arizonapoolfence) that run 8 already found and already correctly logged
  as unverifiable beyond follower count. No new candidate surfaced —
  confirms run 3/4's "WebSearch discovery is structurally biased toward
  indexed/established businesses, not the target profile" finding is
  still true four runs later with fresh queries.

**No prospects added.** Ten consecutive runs, zero verified leads, same
root cause every time: qualifiers 1 (ad evidence) and 2 (funnel dead-end)
both require either the Meta Ad Library link caption or a live IG/FB
profile's bio/link, and nothing reachable from this environment can render
either. This is not going to resolve itself by firing again next hour.

**Recommendation logged here for whoever reads this next (human or
agent):** don't keep re-diagnosing this hourly. Either (a) fix the
proxy-side TLS reset for large ClientHellos so a real browser can pass
Meta's JS challenge, (b) drop the cron frequency drastically (hourly burns
real cost for a guaranteed-identical result until (a) or (c) happens), or
(c) switch inputs entirely — have a human forward candidate ad
screenshots/handles for this session to verify against the business's own
site, which sidesteps the Ad Library/IG wall completely since verification
of qualifier 2 (dead-end) and 3/4 (job value, follower count) can often be
done against the business's own site or a given handle even without Ad
Library access — only qualifier 1 (ad evidence itself) strictly needs
Meta's data if no screenshot is supplied.

Sending a push notification this run (last one was run 5) — the new,
actionable information is the hourly cadence itself plus the confirmed
"still not fixed" status of the one concrete, previously-diagnosed bug.

## 2026-07-29 — Run 9: blocker unchanged; run 8's IG crack did NOT hold

Quick reconfirmation only (full diagnosis already on record in runs 1-8,
not repeated here):

- `facebook.com/ads/library/*` → still HTTP 403, byte-identical JS
  challenge body (`fetch('/__rd_verify_...?challenge=3')`). Unchanged
  across 9 runs now.
- `instagram.com/<handle>` via curl (mobile Safari UA) → **HTTP 302 to the
  login wall this run**, not the HTTP 200 + readable `og:description`
  follower count that run 8 found and confirmed across 3 accounts just
  hours earlier same day. Retried 3x, consistently 302. So that crack was
  either session/IP-specific or patched within hours — **not a stable
  technique, don't rely on it going forward without re-verifying live.**
- `adstransparency.google.com` → still HTTP 200 JS shell (2.5MB), no
  advertiser data reachable without a real JS engine.

No new discovery channel found or attempted this run beyond what runs 1-8
already ruled out (Ad Library, IG API, IG legacy AJAX, TikTok Creative
Center, Yelp, BBB, Maps, Google/Bing/DuckDuckGo SERPs, Wayback). Nothing
in this session's toolset can execute Meta's client-side JS challenge or
render Instagram's client-side-hydrated bio/link data, which is what
qualifiers 1 and 2 require to verify (not fabricate) a lead. No push
notification sent — runs 3 and 5 already escalated this exact blocker with
actionable diagnoses, and nothing has changed that changes what the user
can do about it.

Nine consecutive runs, zero verifiable leads. Zero committed to
inbox.json again this run — recording a guess to avoid an empty file
would violate this brief's core instruction not to fabricate.

## 2026-07-29 — Run 8: found a genuine partial crack (IG follower counts now readable via curl), but it only serves qualifier 4, not 1 or 2 — still zero verifiable leads

Re-tested the three canonical blockers first, per every prior run's own
instruction, using raw `curl` (not just WebFetch, to catch any curl-specific
capability WebFetch doesn't have — this is exactly how run 5 found the
partial 200-status crack in the first place):

- `facebook.com/ads/library/*` → still HTTP 403 via both curl and WebFetch,
  same JS challenge body (`fetch('/__rd_verify_...?challenge=3')`) byte-for-
  byte as runs 5-7. **Unchanged.**
- `instagram.com/api/v1/users/web_profile_info/?username=X` → still HTTP
  400 with the *exact same* Meta-side error string as runs 5-6:
  `"Asset asset://laser.provider/ig_business_category_subvertical has been
  deleted. You cannot use this schema"`. **Unchanged** — three runs running
  now with byte-identical text, confirming this is a static backend
  regression on Meta's end, not worth re-testing again.
- `instagram.com/<handle>/?__a=1&__d=dis` (new this run, the legacy AJAX
  JSON endpoint some scraping guides still reference) → HTTP 201, empty
  body. Not usable.
- `adstransparency.google.com` → still HTTP 200 JS shell. This run went
  further than 2/3/6 and actually grepped the 2.5MB payload for the string
  "advertiser" to check whether real ad records were hiding in an unparsed
  JSON blob — the one hit was Google Ads *policy help-center* boilerplate
  (a link to `support.google.com/adspolicy/...`), not advertiser search
  results. Confirmed empty of real data, not just "probably empty."
- `instagram.com/<handle>` via the **WebFetch tool** → still HTTP 429,
  same as every prior run.

**New this run: `instagram.com/<handle>` via raw `curl` with a mobile
Safari UA reliably returns HTTP 200 AND now contains a real, accurate
`og:description` meta tag with follower/following/post counts** — this is
new information, not just a repeat of run 5's "200 status but shell-only,
zero data" finding. Tested against three real accounts to rule out a
one-off:
- `natgeo` (control, huge account): "269M Followers, 195 Following, 32K
  Posts" — matches Instagram's real public numbers.
- `phoenixpoolfence` (real small Phoenix trade business, found via
  WebSearch for `"DM us for a quote" pool OR fence OR roofing instagram
  Phoenix`): "2,890 Followers, 1,657 Following, 148 Posts" — real number,
  plausible small-account size.
- `arizonapoolfence` (same search): "331 Followers, 222 Following, 114
  Posts."

Then checked how far this actually goes: grepped the full ~680KB response
for `biography`, `bio_links`, `external_url`, `link_in_bio`,
`business_email`, `business_phone`, and any embedded `http(s)://` URL
outside Instagram/Facebook's own CDN domains, across all three accounts and
also dumped every `<script type="application/json">` block. **None of
these fields appear anywhere in the response, on any account.** The page is
Instagram's real hydration bootstrap (Bootloader/ScheduledServerJS payload
defs, CSS/JS resource maps, GraphQL client scaffolding) with the profile's
follower/following/post counts baked into the `og:description` tag for
link-preview purposes, but the bio text and bio link are loaded client-side
after the fact and never appear in the server-sent HTML.

**What this means for the qualifiers, precisely:** this newly-confirmed
`curl` + `og:description` technique is a real, reusable way to verify
qualifier 4 (follower count / "owner-reachable small account") for any
specific handle already in hand — a genuine, if narrow, improvement over
runs 1-7, which had no way to confirm follower counts at all short of
trusting an unverifiable WebSearch snippet. **It does nothing for
qualifiers 1 (ad spend evidence) or 2 (funnel dead-end / ad link caption),**
which is the load-bearing pair every prior run got stuck on — both still
require either the Meta Ad Library (403, unchanged) or the bio/link field
itself (absent from every fetchable surface, unchanged). A verified
follower count with no way to confirm the account is even running ads, or
what its bio link/lack thereof is, cannot pass this brief's "all four must
hold" + "never fabricate" bar on its own.

**Action for future runs:** don't re-derive the og:description finding —
it's confirmed and stable across 3 accounts. Use it opportunistically to
confirm qualifier 4 once a candidate is otherwise sourced, but don't expect
it to unblock sourcing by itself. The real gap is still qualifiers 1/2, and
nothing tried across 8 runs (Ad Library direct, IG profile HTML, IG
internal API, IG legacy AJAX endpoint, Google/TikTok Ads Transparency,
Yelp/BBB/Maps, WebSearch-driven discovery) has found a way to see either an
ad's destination/caption or a profile's bio link from this environment.

Not sending a push notification this run: per runs 6/7's own standard, a
notification is for new *actionable* information, and while the
og:description crack is genuinely new, it doesn't change what the user can
do about the underlying blocker (Ad Library + bio-link access) that runs 3
and 5 already escalated. Logging here so the next run inherits it instead
of re-testing from scratch.

No prospects added this run. Zero committed to inbox.json — eight
consecutive runs now with zero verifiable leads. Qualifiers 1 and 2 remain
unverifiable from this environment; qualifier 4 is now independently
verifiable given a candidate handle, which narrows but does not close the
gap.

## 2026-07-29 — Run 7: reconfirmed the wall again; found the real reason 6 runs of findings never reached the user's repo

Before sourcing, re-tested the three canonical blocked surfaces per every prior
run's own recommendation, both via raw `curl` and via the actual `WebFetch`
tool (not just curl, to rule out a curl-specific artifact):

- `facebook.com/ads/library/*` → HTTP 403, both curl and WebFetch. Identical
  to runs 1-6.
- `instagram.com/<handle>` → curl returned 302 (to login), WebFetch returned
  429. Both non-usable, consistent with the range of outcomes runs 1-6
  reported (302 in early runs, 429 in most, 200-shell-only in run 5/6).
- `adstransparency.google.com` → HTTP 200, JS shell only, no advertiser data.
  Identical to runs 2/3/6.
- Checked `ListConnectors` for a Meta/Instagram/Facebook Business connector
  that might bypass the block entirely — none exists in this account's
  connector list (only Figma, Gmail, Google Calendar, Google Drive,
  higgsfield, Slack, Supabase — none reach Meta ad or profile data).
- Checked the proxy status endpoint (`$HTTPS_PROXY/__agentproxy/status`) —
  `recentRelayFailures` is empty, proxy reports healthy. This doesn't
  contradict run 5's diagnosis (a TLS-reset specific to large/complex
  ClientHellos like Chromium's, not a general outage) but confirms there's
  no obvious proxy-side incident to point to beyond what run 5 already
  found.

**No change in the blocker.** Per run 6's own stated policy, not re-sending
a push notification — run 5 already delivered the specific, actionable
diagnosis (proxy TLS-reset on large ClientHello) and nothing has changed
since. A seventh identical ping would be exactly the noise run 6 warned
against.

**Correction logged in-run, not carried to the commit:** early in this run
`git log --oneline origin/main` (no prior `git fetch` this session) showed
`origin/main` still at the pre-prospecting commit, and `git ls-remote origin
| grep -i prospect` returned nothing — that second check was actually
meaningless (`ls-remote` lists ref/branch names, not commit messages, and no
branch is named "prospect"), so it proved nothing either way. That
combination briefly looked like "6 runs of commits never got pushed."
Running an explicit `git fetch origin main` corrected this: `origin/main`
is already at `e4eb3ba` (run 6's commit) — all 6 prior runs' commits are on
the remote and always were. There was no push bug; the local remote-tracking
ref was just stale until fetched. **Action for future runs:** run `git fetch
origin main` before drawing any conclusion from `git log origin/main` — the
cached ref from session start can be behind the real remote state.

No prospects added this run — the sourcing blocker is unchanged. Seven
consecutive runs now with zero verifiable leads, all for the same
underlying reason (qualifiers 1 and 2 both require Meta Ad Library or live
IG/FB profile access, both unreachable from this environment).

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

## 2026-07-29 — Run 6: reconfirmed run 5's blocker byte-for-byte, tried three new channels (TikTok, Yelp, BBB/Maps), no fix

Re-tested the core surfaces first, per every prior run's own recommendation.
All identical to run 5, down to the exact bytes:

- `facebook.com/ads/library/*` → still HTTP 403 (both raw curl and the
  WebFetch tool).
- `instagram.com/<handle>` via WebFetch → still HTTP 429. Via raw curl with
  a mobile Safari UA → HTTP 200 but (as run 5 found) it's the React shell
  only, no bio/link/follower data server-rendered.
- `instagram.com/api/v1/users/web_profile_info/?username=X` → still fails
  with the *exact same* error string run 5 got:
  `"Asset asset://laser.provider/ig_business_category_subvertical has been
  deleted. You cannot use this schema"`. Identical text two runs running
  confirms this is a static Meta-side backend regression, not something
  that fluctuates — not worth re-testing again short of a Meta-side fix.
- `facebook.com/<small-business-page>` (plain curl, browser UA, tested
  against a real small AZ business page rather than run 5's neutral
  control) → HTTP 302 to login wall, zero bytes. No `og:description`
  fallback available even for a Page (not a profile).
- `adstransparency.google.com` → HTTP 200 shell, no advertiser data (same
  as runs 2/3).

**Three genuinely new channels tried this run, none usable:**
- **TikTok Ad/Creative Center** (`library.tiktok.com/ads`) — not attempted
  by any prior run. Returns a real HTTP 200, 38KB body — but it's a pure
  client-rendered shell (`<div id="root">` empty, all JSON in an obfuscated
  script tag). Same class of failure as Google Ads Transparency Center.
  Also a weaker fit for this brief's target population than Meta anyway —
  small local trade businesses running "boosted post" style spend skew
  heavily Facebook/Instagram, not TikTok — so even if it worked it likely
  wouldn't be the primary channel to build around.
- **Yelp business pages** → HTTP 403 (bot-gated, same class as Facebook).
- **BBB business search** and **Linktree** → both HTTP 200 with real
  static HTML (confirms run 1's Linktree finding still holds). But neither
  carries ad-spend evidence or a bio-link defect on its own — BBB shows
  business existence/category/sometimes a website field, Linktree shows
  destination-page content IF you already have the exact linktr.ee URL
  from somewhere. Neither substitutes for qualifier 1 (ad evidence, which
  only lives in the Meta Ad Library or a visible ad) or qualifier 2's
  specific "ad link caption" requirement (which only lives on the live
  IG/FB profile or ad itself). Google Maps search page also returns 200
  but wasn't pursued further once it was clear it can't carry ad-spend
  evidence either.

**Assessment: this is the same wall as runs 1-5, reconfirmed with zero
drift in failure mode, plus three new negative results.** Nothing in this
session's toolset (WebSearch, WebFetch, raw curl/Bash) can render Meta's
client-side JS or pass its bot challenge, and no alternate primary source
(TikTok, Yelp, BBB, Maps) both (a) is fetchable and (b) carries the
specific evidence qualifiers 1 and 2 require. Not re-sending a push
notification — run 5 already gave the user the specific, actionable
diagnosis (proxy TLS-reset on large ClientHello blocks headless Chrome;
Meta's own backend API is separately broken) and nothing has changed since
to report; a repeat ping with no new information would be noise per this
routine's own standard. Next run should skip re-deriving this and, unless
the environment changes (proxy fix, browser-capable session, or a
user-supplied seed list of candidate handles/ad screenshots per run 4's
suggested workaround), treat sourcing as blocked without spending budget
on it.

No prospects added this run. Zero committed to inbox.json — six
consecutive runs now with zero verifiable leads, all for the same
underlying reason.

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
