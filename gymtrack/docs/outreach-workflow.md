# Gym outreach workflow

Every cold email goes through these steps, in order. Nothing is sent until the checklist for that gym is complete.

## 1. Pick the gym
- [ ] Commercial gym with a real weights floor (not a studio, class-only box or PT-only space).
- [ ] Country without strict workplace/biometric camera rules (UAE, KSA, Qatar, etc. are fine; skip Germany-style regimes).
- [ ] Not already contacted: check Gmail `in:sent` for the domain and the gym name.

## 2. Find the right person (owner first)
Target, in this order:
1. Owner / founder / CEO (independent gyms and small chains).
2. For chains: COO, head of operations, CTO / head of digital or innovation (the person who'd own a camera pilot).
3. A dedicated **partnerships / collaborations / business development** inbox (often the gym's own routing for exactly this kind of proposal).
4. Generic info@/hello@ only as the last resort, and then as a short "who should I speak to?" note, not the full pitch.

Sources to check:
- [ ] Gym site: About, Team, Careers, Press, footer, Terms and Privacy (legal entity + often a named data-protection contact).
- [ ] LinkedIn (company page "People", founder profiles), via search results.
- [ ] Press: Gulf News, Khaleej Times, The National, Arabian Business, Zawya, Entrepreneur ME, Forbes ME, Health Club Management, Club Industry, fitt insider.
- [ ] Franchise/affiliate directories (e.g. CrossFit affiliate listings show owner + contact email).
- [ ] Podcasts, interviews, award listings.

## 3. Find every published address, including partnership inboxes
SEVEN replied that their partnerships go to `collabs@seven.house`: a sister-brand domain, not the gym's own. So:
- [ ] Crawl the gym site AND every related domain for `@` addresses: sister brands, parent group, regional sites (e.g. `seven.club` + `seven.house`, `gigafitdubai.com` + `gigafit.ae` + `gigafit.fr`). Find related domains from the footer, the legal entity, the Instagram bio, and the parent company's site.
- [ ] Search: `"@domain" partnerships OR collab OR collabs OR partners OR bd OR marketing OR press OR media`, and the same for each related domain.
- [ ] Instagram / Facebook / TikTok bio and "Contact" button, Linktree pages, Google Business profile.
- [ ] Press releases' "media contact" lines (these also reveal the email pattern).
- [ ] App store listings (developer contact email).
- [ ] Event, sponsorship and influencer pages ("work with us", "partner with us", "brand collaborations").
If a partnership/collab inbox exists, it goes to the send list (alongside the owner, never instead of a confirmed owner address).

## 4. Work out the email pattern from evidence
- [ ] Collect at least one real named address on the same domain (press contact, staff page, PDF, app store, job ad). Record the URL.
- [ ] Derive the pattern (first@, first.last@, flast@, ...). Data-broker percentages (RocketReach, ZoomInfo masks) are hints only.
- [ ] Check MX (Google Workspace / Microsoft 365 / Zoho) so we know the domain takes mail.
- Confidence:
  - **High**: the exact address is published.
  - **Medium**: pattern proven by a real address on the same domain.
  - **Low**: pattern guessed.

## 5. Verify before sending
- [ ] High / medium: send.
- [ ] Low: verify first with an email verifier (Hunter email verifier or similar, using the founder's account or API key). Do not guess-send.
  Microsoft 365 domains reject unknown recipients outright (`550 5.4.1 Recipient address rejected`, seen on whgym.com), so a wrong guess bounces within minutes.
- [ ] Personal addresses (gmail/outlook) need the founder's explicit OK.
- [ ] One address per person. Never spray first@ / first.last@ / flast@ at the same person.

## 6. Write the email: one idea
The reader should walk away with a single thought:
> Members quit when they can't see progress. Your cameras can show it to them.

- [ ] Subject line states the idea.
- [ ] Opening line is specific to the gym (why them).
- [ ] One short paragraph on the problem (members quitting).
- [ ] One short paragraph on what GymTrack does to solve it (automatic workout logs from existing cameras, progress on the member's phone), plus one sentence on the floor data the owner gets.
- [ ] No feature lists. Don't mention form/posture correction. Never name the product.
- [ ] The ask: no money; we want to develop it together with a few gyms, free, on their existing cameras. Short call or demo.
- [ ] As short as possible: aim for ~70 words, never over 100. Signed Nikita, GymTrack.
- [ ] If routed by another inbox, say who referred us ("Your team at hello@... suggested I get in touch").

## 7. After sending
- [ ] Check Gmail for bounces 10-30 minutes later (`from:mailer-daemon newer_than:1d`). On a bounce, go back to step 4 and log the failed address.
- [ ] Log every send: gym, person, role, address, confidence, source URL, date, result.
- [ ] Replies that route us elsewhere (collab inboxes, a named manager): follow up the same day and add the new route to step 3's list of places to look.
- [ ] Ticket auto-replies (e.g. "[##159952##] Your ticket has been created") mean the email reached a helpdesk. Find a person instead.
- [ ] Follow up once after 5-7 working days if there's no reply.

## Current template (keep it this short)

Never name the product in the email. Sign with first name only.

Subject: Members quit when they can't see their progress

Hi {name},

{one specific line about their gym}

Most members quit the gym because they can't see their progress.

We're developing a program that uses the cameras you already have to track what every member is doing, giving you, your staff and your members far more insight from infrastructure that's already there.

We're still in development and aren't taking on customers, but we'd like to work with you to develop it. If that sounds worth your time, what time would suit you for a call?

Best,
Nikita
