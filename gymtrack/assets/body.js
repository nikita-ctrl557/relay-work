/* GymTrack body module: muscles, exercise mapping, weekly muscle load, the front/back body map,
   training-style archetypes and coaching suggestions.
   Plain browser JS, loaded as a classic script after core.js / ui.js. Every global is prefixed BM_/bm.
   Input shapes:
     set     = {exercise: id|null, machine: id|name, reps, duration_s, start (unix s), weight_kg|null}
     session = {start, end, lifting_s, resting_s, sets}
     history = {sets: [...], sessions: [...], now?: unix s (defaults to the clock)} */
"use strict";

/* ================================================================
   Muscles and exercises
   ================================================================ */
const BM_MUSCLES = [
  {id: "chest",       name: "Chest",        region: "upper", side: "front"},
  {id: "front_delts", name: "Front delts",  region: "upper", side: "front"},
  {id: "side_delts",  name: "Side delts",   region: "upper", side: "both"},
  {id: "rear_delts",  name: "Rear delts",   region: "upper", side: "back"},
  {id: "biceps",      name: "Biceps",       region: "upper", side: "front"},
  {id: "triceps",     name: "Triceps",      region: "upper", side: "back"},
  {id: "forearms",    name: "Forearms",     region: "upper", side: "both"},
  {id: "traps",       name: "Traps",        region: "upper", side: "both"},
  {id: "lats",        name: "Lats",         region: "upper", side: "back"},
  {id: "upper_back",  name: "Upper back",   region: "upper", side: "back"},   // rhomboids + mid traps
  {id: "lower_back",  name: "Lower back",   region: "core",  side: "back"},   // spinal erectors
  {id: "abs",         name: "Abs",          region: "core",  side: "front"},
  {id: "obliques",    name: "Obliques",     region: "core",  side: "front"},
  {id: "glutes",      name: "Glutes",       region: "lower", side: "back"},
  {id: "quads",       name: "Quads",        region: "lower", side: "front"},
  {id: "hamstrings",  name: "Hamstrings",   region: "lower", side: "back"},
  {id: "adductors",   name: "Adductors",    region: "lower", side: "front"},
  {id: "calves",      name: "Calves",       region: "lower", side: "both"},
];
const bmMuscle = id => BM_MUSCLES.find(m => m.id === id);
const bmMuscleName = id => bmMuscle(id)?.name || id;

/* equip: "free" (barbell/dumbbell), "machine" (station/cable), "body" (bodyweight), "cardio".
   Muscle mappings follow the usual prime-mover / synergist split. */
const BM_EXERCISES = {
  // pose-recognised (open floor) ids
  squat:          {name: "Squat",            primary: ["quads", "glutes"],               secondary: ["adductors", "hamstrings", "lower_back", "abs"],                  pattern: "squat",  kind: "compound",  equip: "free"},
  deadlift:       {name: "Deadlift",         primary: ["hamstrings", "glutes", "lower_back"], secondary: ["quads", "traps", "upper_back", "lats", "forearms", "adductors"], pattern: "hinge", kind: "compound", equip: "free"},
  overhead_press: {name: "Overhead press",   primary: ["front_delts", "triceps"],        secondary: ["side_delts", "traps", "upper_back", "abs"],                      pattern: "push",   kind: "compound",  equip: "free"},
  pull_up:        {name: "Pull-up",          primary: ["lats", "biceps"],                secondary: ["upper_back", "rear_delts", "forearms", "abs"],                   pattern: "pull",   kind: "compound",  equip: "body"},
  push_up:        {name: "Push-up",          primary: ["chest", "triceps"],              secondary: ["front_delts", "abs"],                                            pattern: "push",   kind: "compound",  equip: "body"},
  lunge:          {name: "Lunge",            primary: ["quads", "glutes"],               secondary: ["hamstrings", "adductors", "calves"],                             pattern: "lunge",  kind: "compound",  equip: "free"},
  row:            {name: "Bent-over row",    primary: ["upper_back", "lats"],            secondary: ["rear_delts", "biceps", "lower_back", "forearms"],               pattern: "pull",   kind: "compound",  equip: "free"},
  curl:           {name: "Biceps curl",      primary: ["biceps"],                        secondary: ["forearms"],                                                      pattern: "pull",   kind: "isolation", equip: "free"},
  burpee:         {name: "Burpee",           primary: ["quads", "chest"],                secondary: ["glutes", "triceps", "front_delts", "abs", "calves"],             pattern: "conditioning", kind: "conditioning", equip: "body"},
  jumping:        {name: "Box jump / jacks", primary: ["quads", "calves"],               secondary: ["glutes", "hamstrings"],                                          pattern: "conditioning", kind: "conditioning", equip: "body"},
  situp:          {name: "Sit-up",           primary: ["abs"],                           secondary: ["obliques"],                                                      pattern: "core",   kind: "isolation", equip: "body"},
  raise:          {name: "Front raise",      primary: ["front_delts"],                   secondary: ["side_delts", "traps"],                                           pattern: "push",   kind: "isolation", equip: "free"},
  battle_rope:    {name: "Battle ropes",     primary: ["front_delts", "forearms"],       secondary: ["side_delts", "biceps", "abs", "upper_back"],                     pattern: "conditioning", kind: "conditioning", equip: "body"},
  plank:          {name: "Plank / floor core", primary: ["abs"],                         secondary: ["obliques", "front_delts", "glutes"],                             pattern: "core",   kind: "isolation", equip: "body"},
  unknown:        {name: "Unrecognised exercise", primary: [], secondary: [],                                                                                       pattern: null,     kind: null,        equip: null},
  // machine / station exercises
  bench_press:    {name: "Bench press",      primary: ["chest", "triceps"],              secondary: ["front_delts"],                                                   pattern: "push",   kind: "compound",  equip: "free"},
  incline_press:  {name: "Incline press",    primary: ["chest", "front_delts"],          secondary: ["triceps"],                                                       pattern: "push",   kind: "compound",  equip: "free"},
  shoulder_press_machine: {name: "Shoulder press machine", primary: ["front_delts", "triceps"], secondary: ["side_delts", "traps"],                                  pattern: "push",   kind: "compound",  equip: "machine"},
  chest_fly:      {name: "Chest fly",        primary: ["chest"],                         secondary: ["front_delts"],                                                   pattern: "push",   kind: "isolation", equip: "machine"},
  lateral_raise:  {name: "Lateral raise",    primary: ["side_delts"],                    secondary: ["traps"],                                                         pattern: "push",   kind: "isolation", equip: "free"},
  triceps_pushdown: {name: "Triceps pushdown", primary: ["triceps"],                     secondary: [],                                                                pattern: "push",   kind: "isolation", equip: "machine"},
  lat_pulldown:   {name: "Lat pulldown",     primary: ["lats"],                          secondary: ["biceps", "upper_back", "rear_delts"],                            pattern: "pull",   kind: "compound",  equip: "machine"},
  cable_row:      {name: "Cable row",        primary: ["upper_back", "lats"],            secondary: ["rear_delts", "biceps", "forearms"],                              pattern: "pull",   kind: "compound",  equip: "machine"},
  face_pull:      {name: "Face pull",        primary: ["rear_delts", "upper_back"],      secondary: ["traps", "side_delts"],                                           pattern: "pull",   kind: "isolation", equip: "machine"},
  leg_press:      {name: "Leg press",        primary: ["quads", "glutes"],               secondary: ["adductors", "hamstrings"],                                       pattern: "squat",  kind: "compound",  equip: "machine"},
  leg_extension:  {name: "Leg extension",    primary: ["quads"],                         secondary: [],                                                                pattern: "squat",  kind: "isolation", equip: "machine"},
  leg_curl:       {name: "Leg curl",         primary: ["hamstrings"],                    secondary: ["calves"],                                                        pattern: "hinge",  kind: "isolation", equip: "machine"},
  hip_thrust:     {name: "Hip thrust",       primary: ["glutes"],                        secondary: ["hamstrings", "adductors"],                                       pattern: "hinge",  kind: "compound",  equip: "free"},
  woodchop:       {name: "Cable woodchop",   primary: ["obliques"],                      secondary: ["abs"],                                                           pattern: "core",   kind: "isolation", equip: "machine"},
  side_plank:     {name: "Side plank",       primary: ["obliques"],                      secondary: ["abs", "glutes"],                                                 pattern: "core",   kind: "isolation", equip: "body"},
  calf_raise:     {name: "Calf raise",       primary: ["calves"],                        secondary: [],                                                                pattern: "squat",  kind: "isolation", equip: "machine"},
  treadmill:      {name: "Treadmill",        primary: [],                                secondary: ["quads", "calves"],                                               pattern: "conditioning", kind: "conditioning", equip: "cardio"},
  bike:           {name: "Bike",             primary: [],                                secondary: ["quads", "glutes"],                                               pattern: "conditioning", kind: "conditioning", equip: "cardio"},
  rower:          {name: "Rowing erg",       primary: [],                                secondary: ["upper_back", "lats", "quads", "hamstrings"],                     pattern: "conditioning", kind: "conditioning", equip: "cardio"},
};

/* Machine-name keyword rules, first match wins (order matters: "leg press" before "press", "rower" before "row"). */
const BM_MACHINE_RULES = [
  [/incline/, "incline_press"],
  [/bench|chest press/, "bench_press"],
  [/\bfly|flye|pec ?deck/, "chest_fly"],
  [/leg press|hack/, "leg_press"],
  [/leg curl|ham(string)? curl/, "leg_curl"],
  [/leg ext/, "leg_extension"],
  [/calf/, "calf_raise"],
  [/woodchop|wood chop|oblique|twist/, "woodchop"],
  [/hip thrust|glute/, "hip_thrust"],
  [/squat|smith/, "squat"],
  [/dead|hinge|platform/, "deadlift"],
  [/pull ?down|\blat\b|lats/, "lat_pulldown"],
  [/pull[- ]?up|chin/, "pull_up"],
  [/rower|row(ing)? ?erg|\berg\b|concept ?2/, "rower"],
  [/face pull/, "face_pull"],
  [/\brow/, "cable_row"],
  [/curl|bicep/, "curl"],
  [/tricep|push ?down/, "triceps_pushdown"],
  [/lateral|side raise/, "lateral_raise"],
  [/tread|\brun/, "treadmill"],
  [/bike|cycle|spin|airdyne|assault/, "bike"],
  [/shoulder|overhead|military|press/, "shoulder_press_machine"],
];

/* set.exercise if we know it; else infer from the machine name (or id). `machines` (optional: config
   machines [{id, name}]) resolves a machine id to its name first. Floor areas without a pose label → "unknown". */
function bmExerciseFor(set, machines) {
  const ex = set?.exercise;
  if (ex && ex !== "unknown" && BM_EXERCISES[ex]) return ex;
  let name = set?.machine_name || set?.machine || "";
  if (machines && typeof name === "string") { const m = machines.find(x => x.id === name); if (m) name = m.name || name; }
  if (typeof name === "object" && name) name = name.name || name.id || "";
  const n = String(name).toLowerCase().replace(/[_-]+/g, " ");
  for (const [re, id] of BM_MACHINE_RULES) if (re.test(n)) return id;
  return "unknown";
}

/* ================================================================
   Load
   ================================================================ */
/* Weekly target in hard sets per muscle (the common 10-20 sets/week hypertrophy guideline, set at its
   low end: big muscle groups ~10, smaller or heavily-indirectly-trained ones 4-8). */
function bmWeeklyTargets() {
  return {chest: 10, lats: 10, upper_back: 10, quads: 10, hamstrings: 8, glutes: 8,
    front_delts: 6, side_delts: 8, rear_delts: 6, biceps: 8, triceps: 8, forearms: 4, traps: 6,
    lower_back: 6, abs: 8, obliques: 6, adductors: 5, calves: 8};
}

/* Muscle load rule (kept deliberately simple so the UI can explain it as "sets"):
   every set adds 1 to each PRIMARY muscle of its exercise and 0.5 to each SECONDARY muscle.
   Reps do not scale the score: "hard sets" is the standard volume measure, and a 3-rep heavy set
   counts the same as a 12-rep set. Unrecognised exercises add nothing.
   opts.machines → passed to bmExerciseFor. Returns {muscle_id: score} for every muscle (0 if untrained). */
function bmMuscleLoad(sets, opts = {}) {
  const load = Object.fromEntries(BM_MUSCLES.map(m => [m.id, 0]));
  for (const s of sets || []) {
    const ex = BM_EXERCISES[bmExerciseFor(s, opts.machines)];
    for (const m of ex.primary) load[m] += 1;
    for (const m of ex.secondary) load[m] += 0.5;
  }
  return load;
}

const bmNow = h => h?.now ?? Date.now() / 1000;
/* sets in the last `days` days before `now` */
function bmRecentSets(sets, days = 7, now = Date.now() / 1000) { const t0 = now - days * 86400; return (sets || []).filter(s => s.start >= t0 && s.start <= now + 60); }
/* convenience: this week's load for a history */
const bmWeekLoad = (history, opts) => bmMuscleLoad(bmRecentSets(history?.sets, 7, bmNow(history)), opts);

/* ================================================================
   Measured load: what the 3D actually saw, not just the exercise label
   ================================================================ */
/* Full-range reference per exercise for its main 3D angle (degrees, 0 = straight joint): the textbook range of a
   full rep. angle names follow server/gymtrack/pose3d.py (MAIN_ANGLE; lateral / front raises use the shoulder
   angle, which MAIN_ANGLE doesn't list). depth: describe the set by its average bottom-of-rep angle. */
const BM_ROM_REF = {
  squat:          {angle: "knee_flex", full: 120, what: "knee flexion", depth: true},
  lunge:          {angle: "knee_flex", full: 100, what: "knee flexion", depth: true},
  leg_press:      {angle: "knee_flex", full: 100, what: "knee flexion", depth: true},
  deadlift:       {angle: "hip_flex", full: 90, what: "hip hinge"},
  hip_thrust:     {angle: "hip_flex", full: 90, what: "hip hinge"},
  cable_row:      {angle: "elbow_flex", full: 100, what: "elbow"},
  row:            {angle: "elbow_flex", full: 90, what: "elbow"},
  bench_press:    {angle: "elbow_flex", full: 90, what: "elbow"},
  incline_press:  {angle: "elbow_flex", full: 90, what: "elbow"},
  push_up:        {angle: "elbow_flex", full: 90, what: "elbow"},
  overhead_press: {angle: "elbow_flex", full: 100, what: "elbow"},
  shoulder_press_machine: {angle: "elbow_flex", full: 100, what: "elbow"},
  triceps_pushdown: {angle: "elbow_flex", full: 90, what: "elbow"},
  curl:           {angle: "elbow_flex", full: 120, what: "elbow"},
  lat_pulldown:   {angle: "elbow_flex", full: 100, what: "elbow"},
  pull_up:        {angle: "elbow_flex", full: 110, what: "elbow"},
  lateral_raise:  {angle: "shoulder_abd", full: 80, what: "shoulder abduction"},
  raise:          {angle: "shoulder_flex", full: 90, what: "shoulder flexion"},
};
/* a set counts fully for effort at >= 8 reps or >= 30 s under load (each is half of the effort factor) */
const BM_EFFORT = {reps: 8, tul_s: 30, min: 0.25};
const bmClamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* the 3D mobility entry (core.js mobility3d().sets) for one replay set {t0, pid} */
const bmMobFor = (set, mob) => (mob || []).find(c => Math.abs(c.t0 - set.t0) < 0.05 && (c.pid == null || set.pid == null || c.pid === set.pid)) || null;

/* attach 3D mobility entries to history sets ({start, end, reps, exercise}) that have lost their replay times:
   matched in order by exercise, set length (±0.35 s) and reps. Returns new set objects; unmatched sets get m3 = null. */
function bmAttachMeasured(sets, mob, opts = {}) {
  const pool = [...(mob || [])].sort((a, b) => a.t0 - b.t0), used = new Set();
  return [...(sets || [])].sort((a, b) => a.start - b.start).map(s => {
    const ex = bmExerciseFor(s, opts.machines), dur = (s.end ?? s.start) - s.start;
    const c = pool.find(c => !used.has(c) && (!c.ex || c.ex === ex) && Math.abs((c.t1 - c.t0) - dur) < 0.35 && (c.reps_counted == null || c.reps_counted === s.reps));
    if (c) used.add(c);
    return {...s, m3: c || null};
  });
}

/* How much one set counts. s = {exercise|machine, reps, duration_s | start+end, m3: mobility entry | null}.
   measured (3D range of the main angle usable): f = romF × effF, where
     romF = measured range / full-range reference, clamped 0.25..1 (a deeper-than-reference rep doesn't count extra)
     effF = ½·min(1, reps / 8) + ½·min(1, time under load / 30 s), clamped 0.25..1
   otherwise f = 1 (the label-only rule) and est = true, with the reason in note. */
function bmSetFactor(s, opts = {}) {
  const ex = bmExerciseFor(s, opts.machines), ref = BM_ROM_REF[ex], m = s.m3, reps = s.reps || 0;
  const tul = s.duration_s ?? Math.max(0, (s.end ?? s.start) - s.start);
  const out = {ex, reps, tul, f: 1, est: true, note: null, rom: null, ref: ref?.full ?? null, romF: null, effF: null, depth: null};
  if (!m) { out.note = "no 3D for this set"; return out; }
  if (!ref) { out.note = "no full-range reference for this exercise"; return out; }
  const r = m.ranges?.[ref.angle], rom = m.main === ref.angle && m.rom_deg != null ? m.rom_deg : r ? r[1] - r[0] : null;
  if (!(rom > 0) || rom > 175) { out.note = "3D range not usable"; return out; }
  const effF = bmClamp(0.5 * Math.min(1, reps / BM_EFFORT.reps) + 0.5 * Math.min(1, tul / BM_EFFORT.tul_s), BM_EFFORT.min, 1);
  const romF = bmClamp(rom / ref.full, 0.25, 1);
  if (ref.depth) { const pk = m.main === ref.angle ? (m.reps || []).map(x => x.peak).filter(v => v != null) : []; out.depth = pk.length ? bmAvg(pk) : r ? r[1] : null; }
  return {...out, f: romF * effF, est: false, rom, romF, effF};
}

/* Measured muscle load. Per set: f (bmSetFactor) × 1 for each primary muscle, × 0.5 for each secondary one.
   Returns {load: {muscle: effective sets}, detail: {muscle: {sets, eff, n, est, why}}, perSet: [factor...]}, where
   sets = the label-only count (same as bmMuscleLoad), eff = effective sets, n = sets that touched it,
   est = how many of those had no usable 3D, why = a short reason ("Squat ×2, avg depth 92° knee flexion"). */
function bmMeasuredLoad(sets, opts = {}) {
  const load = Object.fromEntries(BM_MUSCLES.map(m => [m.id, 0])), detail = {}, perSet = [];
  for (const s of sets || []) {
    const F = bmSetFactor(s, opts), ex = BM_EXERCISES[F.ex]; perSet.push(F);
    for (const [ids, w] of [[ex.primary, 1], [ex.secondary, 0.5]]) for (const id of ids) {
      const d = detail[id] ||= {sets: 0, eff: 0, n: 0, est: 0, by: {}};
      d.sets += w; d.eff += w * F.f; d.n++; if (F.est) d.est++; load[id] += w * F.f;
      const b = d.by[F.ex] ||= {n: 0, est: 0, eff: 0, rom: [], depth: []};
      b.n++; b.eff += w * F.f; if (F.est) b.est++; else { b.rom.push(F.rom); if (F.depth != null) b.depth.push(F.depth); }
    }
  }
  for (const d of Object.values(detail)) d.why = bmWhy(d.by);
  return {load, detail, perSet};
}
function bmWhy(by) {
  return Object.entries(by).sort((a, b) => b[1].eff - a[1].eff).slice(0, 2).map(([ex, b]) => {
    const E = BM_EXERCISES[ex], ref = BM_ROM_REF[ex], head = `${E?.name || ex} ×${b.n}`;
    if (!b.rom.length) return `${head}, estimated (no 3D)`;
    // sets that differ a lot (a full and a partial squat) show their span: an average would describe neither
    const desc = (vals, word) => { const lo = Math.round(Math.min(...vals)), hi = Math.round(Math.max(...vals));
      return vals.length > 1 && hi - lo > 30 ? `${word} ${lo}–${hi}°` : `${vals.length > 1 ? "avg " : ""}${word} ${Math.round(bmAvg(vals))}°`; };
    const what = ref.depth && b.depth.length ? `${desc(b.depth, "depth")} ${ref.what}` : `${ref.what} ${desc(b.rom, "range")}`;
    return `${head}, ${what}${b.est ? ` (${b.est} estimated)` : ""}`;
  }).join("; ");
}
/* this week's measured load for a history whose sets may carry m3 */
const bmWeekLoadMeasured = (history, opts) => bmMeasuredLoad(bmRecentSets(history?.sets, 7, bmNow(history)), opts);

/* Per-muscle list for a profile: muscle, sets, effective sets vs target, why, "estimated" tag.
   opts.targets: {} to show no targets (bars then scale to the largest); opts.period: "this week" (default) / "this session". */
function bmLoadListHTML(res, opts = {}) {
  const T = opts.targets || bmWeeklyTargets(), period = opts.period || "this week", rows = Object.entries(res?.detail || {}).filter(([, d]) => d.sets > 0).sort((a, b) => b[1].eff - a[1].eff);
  if (!rows.length) return `<div class="bm-ll-empty">No sets ${period}</div>`;
  const maxEff = Math.max(1e-9, ...rows.map(([, d]) => d.eff));
  const anyMeasured = rows.some(([, d]) => d.est < d.n);
  return `<div class="bm-ll" role="table" aria-label="Muscle load ${period}"><div class="bm-ll-h" role="row"><span role="columnheader">Muscle</span><span role="columnheader" class="n">Sets</span><span role="columnheader" class="n">Effective</span></div>`
    + rows.map(([id, d]) => {
      const tag = d.est === d.n ? `<em class="bm-est" title="No usable 3D for these sets: counted from the exercise label">estimated</em>` : d.est ? `<em class="bm-est part" title="${d.est} of ${d.n} sets had no usable 3D">partly estimated</em>` : "";
      const pct = Math.min(100, d.eff / (T[id] || maxEff) * 100);
      return `<div class="bm-ll-r" role="row"><span class="nm" role="cell">${esc(bmMuscleName(id))}</span><span class="n num" role="cell">${bmNum(d.sets)}</span><span class="n num" role="cell"><b>${bmNum(d.eff)}</b>${T[id] ? `<small>/${T[id]}</small>` : ""}</span>`
        + `<span class="bar" aria-hidden="true"><i style="width:${pct.toFixed(1)}%"></i></span><span class="why" role="cell">${esc(d.why)}${tag}</span></div>`;
    }).join("")
    + `</div><p class="bm-ll-note">${anyMeasured ? "Effective sets: each set counts 1 for the muscles it mainly works and ½ for helpers, scaled by the range of motion measured in 3D (against a full-range reference) and by reps and time under load. " : `No 3D measurements ${period}. `}Sets without usable 3D count at face value and are marked estimated.</p>`;
}

/* Muscles clearly under target: load below 35% of the weekly target. Empty when nothing was trained at
   all (no point outlining the whole body). Sorted most-missing first. */
function bmGaps(load, targets = bmWeeklyTargets()) {
  if (!load || !Object.values(load).some(v => v > 0)) return [];
  return BM_MUSCLES.map(m => m.id).filter(id => targets[id] && (load[id] || 0) < 0.35 * targets[id])
    .sort((a, b) => (load[a] || 0) / targets[a] - (load[b] || 0) / targets[b]);
}

/* ================================================================
   Body map (SVG)
   ================================================================ */
/* Every shape is drawn for the figure's LEFT half in a 160×340 view (centre line x = 80) and mirrored,
   so the figure is exactly symmetric. Silhouette = a list of cubic segments from the neck to the crotch. */
const BM_SIL = [[73, 40],
  [72.6, 46, 72.6, 49, 71, 51], [65, 56, 56, 57, 47, 59], [37, 60.5, 30.5, 67, 30, 80],
  [29.5, 96, 27, 112, 26.5, 128], [26, 138, 24, 146, 22.5, 154], [20.5, 168, 18.5, 184, 18.5, 198],
  [18.5, 206, 15.5, 212, 16.5, 222], [17.5, 230, 25, 231, 27, 224], [28.5, 218, 28, 210, 29.5, 202],
  [31.5, 188, 35, 170, 38, 154], [39.5, 146, 42, 140, 43, 132], [44, 120, 44.5, 104, 44.5, 94],
  [46, 112, 49.5, 124, 51, 138], [52, 148, 49.5, 158, 49.5, 170], [49.5, 196, 53, 224, 56.5, 248],
  [57.5, 256, 55.5, 268, 56.5, 282], [57.5, 296, 60.5, 308, 61.5, 318], [61, 324, 56, 328, 56.5, 333],
  [60, 336, 72, 336, 76, 333], [76.5, 328, 74.5, 322, 74.5, 316], [75, 302, 78, 286, 77, 270],
  [76, 262, 75, 254, 75.5, 246], [77.5, 226, 79.5, 204, 80, 186]];
function bmSilPath() {
  const f = n => +n.toFixed(2), [st, ...segs] = BM_SIL, mx = x => f(160 - x);
  let d = `M${st[0]},${st[1]}`;
  for (const s of segs) d += `C${s.join(",")}`;
  // mirrored half, walked backwards: each segment's controls swap, endpoint = previous segment's end
  const pts = [st, ...segs.map(s => [s[4], s[5]])];
  for (let i = segs.length - 1; i >= 0; i--) { const s = segs[i], p = pts[i]; d += `C${mx(s[2])},${s[3]},${mx(s[0])},${s[1]},${mx(p[0])},${p[1]}`; }
  return d + "Z";
}

/* [muscle id, [paths], [seam paths]] in draw order (later shapes sit on top). */
const BM_FRONT = [
  ["traps",       ["M72.5,48C70,54 61,57 52,59.5C60,61.5 68,61 76,57.5C74.5,54.5 73.4,51.5 72.5,48Z"]],
  ["chest",       ["M78.5,63C70,61.6 60,62 51.5,64.6C48.6,70 46.8,78 45.8,86C45.3,92 46.2,97 49,100.5C56,106.5 66,107.5 74,104.5C77,103.3 78.5,100.5 78.5,96Z"]],
  ["front_delts", ["M50,60C41,60 33,65 31.6,75C30.8,85 32.4,95 36,103.2C39.4,98 43,90 46.4,82C49,76 51.6,66 50,60Z"]],
  ["side_delts",  ["M44.5,60.3C37,61.3 32.4,66.5 31.6,75C30.8,85 32.4,95 36,103.2C36.2,92 36.6,80 38.6,72C40,66.6 42,63 44.5,60.3Z"]],
  ["biceps",      ["M38.4,104.5C34.8,112 32.3,123 32.2,133.5C33.4,140.5 38.8,141.5 40.8,135.5C42.7,126 43.5,115 42.6,105.5C41.4,103.6 39.6,103.4 38.4,104.5Z"]],
  ["forearms",    ["M24.6,150C22,164 20.6,180 21,194C23.4,197.2 27,197.2 29,194C31,180 34,166 37.1,152C33,147.4 28,147.2 24.6,150Z"]],
  ["obliques",    ["M69.3,107.5C63,106.5 55.5,104.5 49.8,102.5C50.8,114 52.6,127 53.1,139C53.2,150 54.8,160 59.2,168C63.2,168.6 67,163 68.6,154.5C68,139 68.3,123 69.3,107.5Z"],
                  ["M51.5,115C56,113 61,113 66,116", "M53,128C57,126 62,126.5 67,129.5"]],
  ["abs",         ["M78.5,107.6L71.6,107.2C70.2,122 69.7,140 70.2,156C71.1,166 74.2,174 78.5,180.5Z"],
                  ["M70.5,122.5C73.5,121.5 76,121.6 78.5,122.5", "M70.1,138C73.2,137 76,137 78.5,138", "M70.3,154C73.2,153.3 76,153.4 78.5,154.3"]],
  ["quads",       ["M55,174C59,168 66,169 71.2,176.5C74.2,198 75.2,222 74.7,242C70.4,248.4 62.4,248.4 59,242.4C56,222 53.6,196 55,174Z"],
                  ["M64.5,178C64,200 65.5,222 67.5,243", "M74.6,224C69.5,228 67,235 67.5,243"]],
  ["adductors",   ["M72.8,177.5C75.8,180.5 78.6,185 79.1,191C78.7,204 77.2,216 75.3,226C74.7,208 74.2,192 72.8,177.5Z"]],
  ["calves",      ["M58.3,262C56.8,276 58.2,292 61.2,306C62.6,308.2 64.2,306.4 64.2,302C63.7,288 62.7,274 62,262.5C61,260.6 59.3,260.6 58.3,262Z",
                   "M73.4,262.5C76.2,272 76.3,288 73.6,304C72.6,306.4 71.1,305 71,300.5C71.4,288 71.6,274 70.8,263C71.5,261 72.6,261 73.4,262.5Z"]],
];
const BM_BACK = [
  ["lats",        ["M56,88C52,90 48.5,93 46.6,97.4C48.6,112 51.6,126 53.6,142.5C60,146.5 70,143.5 78.5,136.5L78.5,128C70,118 62,104 56,88Z"]],
  ["upper_back",  ["M78.5,91C77,85 74.5,79 70.5,74.5C64,70 57,66.5 51.5,64.5C51,72 52.5,80 56,88C61,100 69,112 78.5,125Z"]],
  ["traps",       ["M73.2,43.5C73.2,49 71.2,52 67,54.2C61,57 54.5,58.6 49,60.2C57,63.2 66,67.5 72.2,73.5C75.2,77.4 77.6,82.5 78.5,88L78.5,45.5C76.6,44.5 75,44 73.2,43.5Z"]],
  ["lower_back",  ["M78.5,128.5C75.2,124.6 72.4,121.6 70.2,120.4C68.7,134 68.1,150 69.1,162.5C71.6,168 75.1,169.6 78.5,169.8Z"]],
  ["rear_delts",  ["M50,60C41,60 33,65 31.6,75C30.8,85 32.4,95 36,103.2C39.4,98 43,90 46.4,82C49,76 51.6,66 50,60Z"]],
  ["side_delts",  ["M44.5,60.3C37,61.3 32.4,66.5 31.6,75C30.8,85 32.4,95 36,103.2C36.2,92 36.6,80 38.6,72C40,66.6 42,63 44.5,60.3Z"]],
  ["triceps",     ["M38.6,103.4C34.2,110.5 31.2,122 30.7,134C33,141 39.2,142.2 41.6,136.2C43.6,126 44.1,114 43.1,104.4C41.6,102.6 39.8,102.4 38.6,103.4Z"],
                  ["M36.6,108C38,116 39.5,124 41.5,131"]],
  ["forearms",    ["M24.6,150C22,164 20.6,180 21,194C23.4,197.2 27,197.2 29,194C31,180 34,166 37.1,152C33,147.4 28,147.2 24.6,150Z"]],
  ["glutes",      ["M78.5,168.4C72,160.4 62,160.2 55,166.2C51,174 51,188 55,198.4C62,205.4 72,204.6 78.5,198.4Z"]],
  ["hamstrings",  ["M55.6,204.4C55.6,222 57.6,238 60.1,248.4C64.1,252.4 70.1,252.4 74.6,246.4C75.6,230 77.6,216 78.6,205.2C70,209.4 62,209.4 55.6,204.4Z"],
                  ["M67.2,209.5C66.8,224 67,238 67.6,251"]],
  ["calves",      ["M58.6,258C55.6,270 56.6,284 60.6,295.4C63.1,300.2 66.1,298.4 66.6,290L66.6,262C64.1,258.2 61.1,256.4 58.6,258Z",
                   "M68.1,262C70.6,257.4 74.1,257.4 75.4,262C77,276 75.6,290 72,298C69.6,300.4 68.1,296.4 68.1,290Z"]],
];

const bmNum = v => { const r = Math.round(v * 10) / 10; return String(r).replace(/\.0$/, ""); };
/* sequential single-hue scale: 0 → neutral; otherwise 24% → 100% plate blue as load approaches the target */
function bmFill(ratio) {
  if (!(ratio > 0)) return "var(--surface-3)";
  return `color-mix(in srgb, var(--lift) ${Math.round(24 + 76 * Math.min(1, ratio))}%, var(--surface-3))`;
}
const BM_MIRROR = `transform="matrix(-1 0 0 1 160 0)"`;

/* bmBodyMap(load, opts) → SVG string (width:100%, scales with its container).
   opts.view: "both" (default) | "front" | "back"; opts.highlight: muscle ids to outline (dashed, orange);
   opts.targets: override bmWeeklyTargets(); opts.legend: false to omit the in-SVG legend;
   opts.label: aria label prefix; opts.unit: tooltip unit (default "sets this week"); opts.notes: {muscle: extra tooltip line}. */
function bmBodyMap(load = {}, opts = {}) {
  const view = opts.view || "both", targets = opts.targets || bmWeeklyTargets(), unit = opts.unit || "sets this week";
  const hl = new Set(opts.highlight || []), legend = opts.legend !== false;
  const views = view === "front" ? ["front"] : view === "back" ? ["back"] : ["front", "back"];
  const VW = 160, GAP = 24, W = views.length * VW + (views.length - 1) * GAP, H = legend ? 392 : 362;
  const sil = bmSilPath();
  const tipFor = id => `${bmMuscleName(id)}\n${bmNum(load[id] || 0)} ${unit}${targets[id] ? ` · target ${targets[id]}` : ""}${opts.notes?.[id] ? "\n" + opts.notes[id] : ""}`;
  const aria = (opts.label || "Muscles trained this week") + ": " + BM_MUSCLES.map(m => `${m.name} ${bmNum(load[m.id] || 0)}${targets[m.id] ? " of " + targets[m.id] : ""} sets`).join(", ");
  let s = `<svg class="bm-map" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${esc(aria)}" style="display:block;width:100%;height:auto;max-width:${Math.round(W * 1.35)}px;margin:0 auto;font-family:var(--f-display)">`
    + `<title>${esc(opts.label || "Muscles trained this week")}</title>`
    + `<style>.bm-map .bm-r{cursor:default;transition:filter .15s}.bm-map .bm-r:hover{filter:brightness(1.12) saturate(1.1)}.bm-map .bm-r:hover path:not(.bm-seam){stroke:var(--ink-2)}`
    + `.bm-map .bm-seam{fill:none;stroke:var(--surface);stroke-width:1.1;stroke-linecap:round;opacity:.75;pointer-events:none}</style>`;
  views.forEach((v, i) => {
    const ox = i * (VW + GAP);
    s += `<g transform="translate(${ox},0)">`;
    // silhouette: body, neck, head
    s += `<g style="fill:var(--surface-2);stroke:var(--line-2);stroke-width:1.3;stroke-linejoin:round"><path d="${sil}"/><ellipse cx="80" cy="26" rx="12.6" ry="15.6"/></g>`;
    if (v === "back") s += `<path d="M67.8,30C67.2,17 73,10.8 80,10.8C87,10.8 92.8,17 92.2,30C88,33 84,34 80,34C76,34 72,33 67.8,30Z" style="fill:var(--surface-3);pointer-events:none"/>`;
    else s += `<path d="M75,52C77,54 83,54 85,52" style="fill:none;stroke:var(--line-2);stroke-width:1;stroke-linecap:round"/>`;
    // knee caps (subtle landmarks)
    s += `<g style="fill:none;stroke:var(--line-2);stroke-width:1;stroke-linecap:round;opacity:.9"><path d="M61,252C64,256 70,256 73,252"/><path d="M99,252C96,256 90,256 87,252" /></g>`;
    for (const [id, paths, seams] of v === "front" ? BM_FRONT : BM_BACK) {
      const r = targets[id] ? (load[id] || 0) / targets[id] : (load[id] > 0 ? 1 : 0);
      s += `<g class="bm-r" data-muscle="${id}" data-tip="${esc(tipFor(id))}" style="fill:${bmFill(r)};stroke:var(--surface);stroke-width:1.15;stroke-linejoin:round">`;
      for (const d of paths) s += `<path d="${d}"/><path d="${d}" ${BM_MIRROR}/>`;
      for (const d of seams || []) s += `<path class="bm-seam" d="${d}"/><path class="bm-seam" d="${d}" ${BM_MIRROR}/>`;
      s += `</g>`;
    }
    // gap outlines on top of everything
    for (const [id, paths] of v === "front" ? BM_FRONT : BM_BACK) if (hl.has(id))
      for (const d of paths) s += `<path d="${d}" style="fill:none;stroke:var(--rest);stroke-width:1.4;stroke-dasharray:3 2.2;pointer-events:none"/><path d="${d}" ${BM_MIRROR} style="fill:none;stroke:var(--rest);stroke-width:1.4;stroke-dasharray:3 2.2;pointer-events:none"/>`;
    s += `<text x="80" y="358" text-anchor="middle" style="fill:var(--muted);font-size:12.5px;font-weight:600;letter-spacing:.16em">${v === "front" ? "FRONT" : "BACK"}</text>`;
    s += `</g>`;
  });
  if (legend) {
    const items = [["None", bmFill(0)], ["Some", bmFill(0.5)], ["On target", bmFill(1)]];
    if (hl.size) items.push(["Under target", null]);
    const widths = items.map(([t]) => 18 + t.length * 5.6 + 12), total = widths.reduce((a, b) => a + b, 0) - 12;
    let x = (W - total) / 2;
    s += `<g style="font-family:var(--f-body);font-size:10.5px">`;
    items.forEach(([t, f], k) => {
      s += f ? `<rect x="${x}" y="376" width="12" height="9" rx="2.5" style="fill:${f}"/>`
        : `<rect x="${x + .7}" y="376.7" width="10.6" height="7.6" rx="2" style="fill:none;stroke:var(--rest);stroke-width:1.3;stroke-dasharray:3 2"/>`;
      s += `<text x="${x + 17}" y="384.4" style="fill:var(--ink-2)">${t}</text>`;
      x += widths[k];
    });
    s += `</g>`;
  }
  return s + `</svg>`;
}

/* The same legend as HTML, for layouts that put it outside the SVG (use bmBodyMap(load, {legend:false})). */
function bmLegend(withGap = false) {
  const sw = (bg, extra = "") => `<i style="display:inline-block;width:12px;height:9px;border-radius:2.5px;background:${bg};${extra}"></i>`;
  const item = (h, t) => `<span style="display:inline-flex;align-items:center;gap:6px">${h}${t}</span>`;
  return `<div class="bm-legend" style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:11.5px;color:var(--ink-2)">`
    + item(sw(bmFill(0)), "None") + item(sw(bmFill(0.5)), "Some") + item(sw(bmFill(1)), "On target")
    + (withGap ? item(sw("transparent", "border:1.3px dashed var(--rest);box-sizing:border-box"), "Under target") : "") + `</div>`;
}

/* ================================================================
   Features, archetypes, coaching
   ================================================================ */
const bmFmtRest = s => { s = Math.round(s || 0); return s < 60 ? s + " s" : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
const bmPct = v => Math.round((v || 0) * 100) + "%";
const bmAvg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;

/* group sets into workouts: a gap of more than 45 min starts a new one */
function bmWorkouts(sets) {
  const ss = [...(sets || [])].sort((a, b) => a.start - b.start), out = [];
  for (const s of ss) { const w = out[out.length - 1]; if (w && s.start - w.end <= 2700) { w.sets.push(s); w.end = Math.max(w.end, s.start + (s.duration_s || 0)); } else out.push({start: s.start, end: s.start + (s.duration_s || 0), sets: [s]}); }
  return out;
}
/* rests: gaps between the end of one set and the start of the next, counted when 0 < gap ≤ 10 min
   (walking to the next machine is rest too; longer gaps are breaks, not rest) */
function bmRests(sets) {
  const ss = [...(sets || [])].sort((a, b) => a.start - b.start), r = [];
  for (let i = 1; i < ss.length; i++) { const g = ss[i].start - (ss[i - 1].start + (ss[i - 1].duration_s || 0)); if (g > 0 && g <= 600) r.push({gap: g, after: ss[i - 1]}); }
  return r;
}

/* Features over the whole history. Shares are 0..1; null when there is no data to judge. */
function bmFeatures(history = {}, opts = {}) {
  const sets = history.sets || [], ids = sets.map(s => bmExerciseFor(s, opts.machines));
  const known = ids.map(id => BM_EXERCISES[id]).filter(e => e.pattern);
  const lifts = sets.filter((s, i) => { const e = BM_EXERCISES[ids[i]]; return e.kind === "compound" || e.kind === "isolation"; });
  const reps = lifts.map(s => s.reps).filter(r => r > 0);
  const rests = bmRests(sets).map(r => r.gap);
  const cnt = f => known.filter(f).length, share = f => known.length ? cnt(f) / known.length : null;
  const comp = cnt(e => e.kind === "compound"), iso = cnt(e => e.kind === "isolation");
  const load = bmMuscleLoad(sets, opts);
  const reg = {upper: 0, lower: 0, core: 0}; for (const m of BM_MUSCLES) reg[m.region] += load[m.id];
  const regT = reg.upper + reg.lower + reg.core;
  const push = cnt(e => e.pattern === "push"), pull = cnt(e => e.pattern === "pull");
  const workouts = bmWorkouts(sets);
  let sessions = history.sessions?.length ? history.sessions : workouts.map(w => ({start: w.start, end: w.end, lifting_s: w.sets.reduce((a, s) => a + (s.duration_s || 0), 0)}));
  const t = sets.map(s => s.start).concat(sessions.map(s => s.start)), span = t.length ? (Math.max(...t) - Math.min(...t)) / 86400 : 0;
  const weeks = Math.max(1, (span + 1) / 7);
  const durs = sessions.map(s => (s.end - s.start) / 60).filter(d => d > 0);
  const work = sessions.reduce((a, s) => a + (s.lifting_s || 0), 0), total = sessions.reduce((a, s) => a + Math.max(0, s.end - s.start), 0);
  return {
    n_sets: sets.length, n_known: known.length, unknown_share: sets.length ? 1 - known.length / sets.length : null,
    avg_reps: bmAvg(reps), avg_rest_s: bmAvg(rests),
    compound_share: comp + iso ? comp / (comp + iso) : null,
    isolation_share: comp + iso ? iso / (comp + iso) : null,
    machine_share: share(e => e.equip === "machine"),
    free_compound_share: share(e => e.kind === "compound" && (e.equip === "free" || e.equip === "body")),
    conditioning_share: share(e => e.kind === "conditioning"),
    split: regT ? {upper: reg.upper / regT, lower: reg.lower / regT, core: reg.core / regT} : null,
    push_sets: push, pull_sets: pull, push_pull: pull ? push / pull : push ? Infinity : null,
    sessions: sessions.length, sessions_per_week: sessions.length / weeks,
    patterns_per_session: bmAvg(workouts.map(w => new Set(w.sets.map(s => BM_EXERCISES[bmExerciseFor(s, opts.machines)].pattern).filter(Boolean)).size)),
    avg_session_min: bmAvg(durs), work_ratio: total ? work / total : null,
  };
}

const BM_ARCHETYPES = {
  strength:     {name: "Strength builder",             summary: "Heavy, low-rep compound lifts with long recoveries: training to move more weight."},
  hypertrophy:  {name: "Hypertrophy / bodybuilder",    summary: "Moderate reps, moderate rests and plenty of machine and isolation work: training for muscle size."},
  conditioning: {name: "Conditioning / CrossFit-style", summary: "Short rests, high reps and many movement patterns per session: training for work capacity."},
  machine:      {name: "Machine-first beginner",       summary: "Mostly guided machines in short sessions; free-weight compounds are the next step."},
  upper:        {name: "Upper-body focused",           summary: "Most of the work goes to chest, back and arms; legs get comparatively little."},
  legs:         {name: "Leg-day loyalist",             summary: "Lower body takes the majority of the volume."},
  balanced:     {name: "Balanced all-rounder",         summary: "No single extreme: a mix of rep ranges, patterns and body regions."},
  new:          {name: "Just getting started",         summary: "Not enough sets yet to read a training style; a few more sessions will tell."},
};

/* Rule-based archetype. Style archetypes (strength, conditioning, machine, hypertrophy) score the share of
   their 3 conditions that hold (conditions without data are skipped; at least 2 must be judgeable).
   Focus archetypes (upper, legs) score 0.75 when their single condition holds, so a fully matching style
   (1.0) beats them but a 2-of-3 style (0.67) does not. Best score ≥ 0.66 wins, ties go by list order;
   otherwise "Balanced all-rounder". Fewer than 8 recognised sets → "Just getting started". */
function bmArchetype(history = {}, opts = {}) {
  const f = bmFeatures(history, opts);
  const pick = (id, traits, score) => ({id, ...BM_ARCHETYPES[id], traits, features: f, score});
  if (f.n_known < 8) return pick("new", [`${f.n_sets} sets recorded so far`], 1);
  const R = f.avg_reps, rest = f.avg_rest_s, low = f.split?.lower;
  const c = (val, ok, yes) => val == null ? null : {ok: !!ok, yes};
  const rules = {
    strength: [c(R, R <= 5, `Low reps: ${bmNum(R)} per set on average`), c(rest, rest >= 150, `Long rests: ${bmFmtRest(rest)} between sets`), c(f.compound_share, f.compound_share >= 0.7, `${bmPct(f.compound_share)} of lifts are compounds`)],
    conditioning: [c(rest, rest < 60, `Short rests: ${bmFmtRest(rest)} between sets`), c(R ?? f.conditioning_share, (R != null && R >= 12) || f.conditioning_share >= 0.25, R >= 12 ? `High reps: ${bmNum(R)} per set` : `${bmPct(f.conditioning_share)} conditioning work`), c(f.patterns_per_session, f.patterns_per_session >= 4, `${bmNum(f.patterns_per_session)} movement patterns per session`)],
    machine: [c(f.machine_share, f.machine_share >= 0.7, `${bmPct(f.machine_share)} of sets on machines`), c(f.free_compound_share, f.free_compound_share < 0.25, `Few free-weight compounds (${bmPct(f.free_compound_share)})`), c(f.avg_session_min, f.avg_session_min < 70, `Short sessions: ${Math.round(f.avg_session_min)} min`)],
    hypertrophy: [c(R, R >= 6 && R <= 12, `Moderate reps: ${bmNum(R)} per set`), c(rest, rest >= 60 && rest <= 150, `Moderate rests: ${bmFmtRest(rest)}`), c(f.machine_share ?? f.isolation_share, Math.max(f.machine_share || 0, f.isolation_share || 0) >= 0.35, `${bmPct(Math.max(f.machine_share || 0, f.isolation_share || 0))} machine / isolation work`)],
    upper: [c(low, low < 0.25, `Only ${bmPct(low)} of the work is lower body`)],
    legs: [c(low, low > 0.6, `${bmPct(low)} of the work is lower body`)],
  };
  let best = null;
  for (const [id, conds] of Object.entries(rules)) {
    const j = conds.filter(Boolean); if (!j.length || (conds.length > 1 && j.length < 2)) continue;
    const met = j.filter(x => x.ok), score = conds.length === 1 ? (met.length ? 0.75 : 0) : met.length / j.length;
    if (!best || score > best.score) best = {id, score, traits: met.map(x => x.yes)};
  }
  const general = [];
  if (f.split) general.push(`Split ${bmPct(f.split.upper)} upper · ${bmPct(f.split.lower)} lower · ${bmPct(f.split.core)} core`);
  if (f.sessions) general.push(`${bmNum(f.sessions_per_week)} sessions per week`);
  if (best && best.score >= 0.66) return pick(best.id, [...best.traits, ...general], best.score);
  const traits = [];
  if (R != null) traits.push(`Mixed rep ranges: ${bmNum(R)} per set on average`);
  if (f.push_pull != null && isFinite(f.push_pull)) traits.push(`Push:pull ${bmNum(f.push_pull)}:1`);
  return pick("balanced", [...traits, ...general], best?.score || 0);
}

/* exercises that train a muscle as PRIMARY, best candidates first (compound free-weight / machine before
   isolation, never conditioning), optionally preferring ones the member already does */
function bmExercisesFor(muscle, n = 2, prefer = [], avoid = new Set()) {
  const ok = ([id, e]) => e.kind !== "conditioning" && !e.primary.some(m => avoid.has(m) && m !== muscle);
  let c = Object.entries(BM_EXERCISES).filter(([id, e]) => e.primary.includes(muscle) && ok([id, e]));
  if (!c.length) c = Object.entries(BM_EXERCISES).filter(([id, e]) => e.secondary.includes(muscle) && ok([id, e]));
  const rank = ([id, e]) => (prefer.includes(id) ? -10 : 0) + (e.kind === "compound" ? 0 : 1) + Math.max(0, e.primary.indexOf(muscle)) * 0.5;
  return c.sort((a, b) => rank(a) - rank(b)).slice(0, n).map(([id]) => id);
}
const bmExName = id => BM_EXERCISES[id]?.name || id;
const bmOr = ids => [...new Set(ids)].map(bmExName).join(" or ");
const bmSets = n => `${bmNum(n)} set${bmNum(n) === "1" ? "" : "s"}`;
const bmIsAre = id => /s$/.test(bmMuscleName(id)) ? "are" : "is";

/* Coaching notes → [{title, detail, level: "warn"|"tip"|"good"}], warnings first.
   load: this week's bmMuscleLoad (defaults to the last 7 days of history.sets). */
function bmRecommendations(history = {}, load, opts = {}) {
  const f = bmFeatures(history, opts), T = bmWeeklyTargets(), now = bmNow(history);
  load = load || bmWeekLoad(history, opts);
  const out = [], add = (level, title, detail) => out.push({level, title, detail});
  if (!f.n_sets) return [{level: "tip", title: "No sets recorded yet", detail: "Train a couple of sessions in front of the cameras and your muscle map and coaching notes appear here."}];
  const done = [...new Set((history.sets || []).map(s => bmExerciseFor(s, opts.machines)))];
  const covered = new Set();

  // push / pull balance
  if (f.push_pull != null && f.push_sets + f.pull_sets >= 6) {
    if (f.push_pull > 1.5) add("warn", "Add pulling: rows or pull-ups", `${f.push_sets} pushing sets vs ${f.pull_sets} pulling (${isFinite(f.push_pull) ? bmNum(f.push_pull) + ":1" : "no pulling at all"}). Aim for at least one pull per push — e.g. ${bmOr(bmExercisesFor("upper_back", 1, done).concat(bmExercisesFor("lats", 1, done.filter(x => !bmExercisesFor("upper_back", 1, done).includes(x)))))} — to keep shoulders healthy and posture balanced.`);
    else if (f.push_pull < 0.67) add("tip", "Add some pressing", `${f.pull_sets} pulling sets vs ${f.push_sets} pushing. Balance it with ${bmOr(bmExercisesFor("chest", 2, done))}.`);
    else add("good", "Push and pull are balanced", `${f.push_sets} pushing vs ${f.pull_sets} pulling sets (${bmNum(f.push_pull)}:1).`);
  }
  // posterior chain / core missing this week
  const wk = m => load[m] || 0;
  if (wk("hamstrings") + wk("glutes") + wk("lower_back") < 3) {
    add("warn", "No posterior chain work this week", `Hamstrings, glutes and lower back got ${bmSets(wk("hamstrings") + wk("glutes") + wk("lower_back"))} combined. Add ${bmOr(["deadlift", "hip_thrust", "leg_curl"].slice(0, 2))} — they protect knees and back and carry over to every lift.`);
    ["hamstrings", "glutes", "lower_back"].forEach(m => covered.add(m));
  }
  if (wk("abs") + wk("obliques") < 2) {
    add("tip", "Give your core some direct work", `Abs and obliques got ${bmSets(wk("abs") + wk("obliques"))} this week. Two or three sets of planks at the end of a session is enough to start.`);
    ["abs", "obliques"].forEach(m => covered.add(m));
  }
  // under-target muscles, most missing first
  const gaps = bmGaps(load, T).filter(m => !covered.has(m) && m !== "forearms");
  gaps.slice(0, 3).forEach(m => add("tip", `${bmMuscleName(m)} ${bmIsAre(m)} under target`, `${bmNum(wk(m))} of ~${T[m]} sets this week. Try ${bmOr(bmExercisesFor(m, 2, done))}.`));
  if (gaps.length > 3) add("tip", "Also light this week", gaps.slice(3).map(bmMuscleName).join(", ") + ".");
  // rest length vs the goal the rep range implies
  const rests = bmRests(history.sets || []);
  const kindOf = st => BM_EXERCISES[bmExerciseFor(st, opts.machines)].kind, lifting = k => k === "compound" || k === "isolation";
  const hyp = rests.filter(r => r.after.reps >= 6 && r.after.reps <= 15 && lifting(kindOf(r.after))).map(r => r.gap);
  const str = rests.filter(r => r.after.reps > 0 && r.after.reps <= 5 && kindOf(r.after) === "compound").map(r => r.gap);
  const hypMain = hyp.length >= 0.4 * rests.filter(r => lifting(kindOf(r.after))).length;
  if (hyp.length >= 5 && hypMain && bmAvg(hyp) > 150) add("warn", "Rests are longer than your sets need", `Your rests average ${bmFmtRest(bmAvg(hyp))} on hypertrophy sets (6–15 reps); 60–90 s fits that goal and fits more work into the same visit.`);
  else if (hyp.length >= 5 && hypMain && bmAvg(hyp) >= 45 && bmAvg(hyp) <= 150) add("good", "Rest times fit your rep range", `${bmFmtRest(bmAvg(hyp))} between moderate-rep sets — inside the useful 1–2½ min window for that rep range.`);
  if (str.length >= 5 && bmAvg(str) < 120) add("tip", "Rest longer between heavy sets", `Heavy sets (≤5 reps) are followed by ${bmFmtRest(bmAvg(str))} of rest on average; 2–5 min lets you keep the weight up across sets.`);
  // frequency
  if (f.sessions_per_week < 1.6) add("tip", "Train a little more often", `${bmNum(f.sessions_per_week)} sessions per week. Two or three lets each muscle get trained twice a week, which grows strength faster than once.`);
  else if (f.sessions_per_week > 6) add("tip", "Plan a rest day", `${bmNum(f.sessions_per_week)} sessions per week. One or two full rest days help recovery keep up with the work.`);
  else add("good", "Consistent training", `${bmNum(f.sessions_per_week)} sessions per week — a frequency that builds results.`);
  // long, low-density sessions
  if (f.avg_session_min > 90 && f.work_ratio != null && f.work_ratio < 0.2) add("warn", "Long sessions, little time under load", `Sessions average ${Math.round(f.avg_session_min)} min but only ${bmPct(f.work_ratio)} of that is lifting. Shorter rests or supersets would fit the same work into ~60 min.`);
  // good notes
  if (!gaps.length && !covered.size) add("good", "Every muscle group was trained this week", "Nothing is under target — keep the plan.");
  const hit = BM_MUSCLES.filter(m => T[m.id] && wk(m.id) >= T[m.id]).map(m => m.name);
  if (hit.length && hit.length < BM_MUSCLES.length) add("good", `On target: ${hit.slice(0, 4).join(", ")}${hit.length > 4 ? ` +${hit.length - 4}` : ""}`, "These reached the weekly set target.");
  if (f.compound_share >= 0.6 && f.n_known >= 10) add("good", "Built on compound lifts", `${bmPct(f.compound_share)} of lifting sets are multi-joint movements — the most efficient use of gym time.`);
  const order = {warn: 0, tip: 1, good: 2};
  return out.sort((a, b) => order[a.level] - order[b.level]);
}

/* training groups used to name a focus */
const BM_GROUPS = {
  Pull: ["lats", "upper_back", "rear_delts", "biceps"],
  Push: ["chest", "front_delts", "side_delts", "triceps"],
  Legs: ["quads", "glutes", "adductors", "calves"],
  "Posterior chain": ["hamstrings", "glutes", "lower_back"],
  Core: ["abs", "obliques"],
};
/* What to train next. Muscles trained in the last 48 h (≥2 sets' worth of load) are rested. Each muscle's
   need = how far below this week's target it is (0..1) × √target (big muscles weigh more). The group (Pull,
   Push, Legs, Posterior chain, Core) with the highest mean need over its rested muscles is the main focus;
   the neediest muscle outside that group is added ("Pull + hamstrings"). Exercises: one for each of the
   group's two neediest muscles plus one for the extra muscle, preferring ones the member already does and
   never one whose prime mover is being rested.
   → {focus, exercises:[ids], muscles:[ids], avoid:[ids], reason} */
function bmNextSession(history = {}, load, opts = {}) {
  const T = bmWeeklyTargets(), now = bmNow(history);
  load = load || bmWeekLoad(history, opts);
  const recent = bmMuscleLoad(bmRecentSets(history.sets, 2, now), opts);
  const avoid = new Set(Object.keys(recent).filter(m => recent[m] >= 2));
  const done = [...new Set((history.sets || []).map(s => bmExerciseFor(s, opts.machines)))];
  const need = m => Math.max(0, 1 - (load[m] || 0) / T[m]) * Math.sqrt(T[m]);
  const groups = Object.entries(BM_GROUPS).map(([g, ms]) => { const ok = ms.filter(m => !avoid.has(m)).sort((a, b) => need(b) - need(a)); return {g, ok, score: ok.length ? ok.reduce((x, m) => x + need(m), 0) / ms.length : -1}; })
    .filter(x => x.ok.length && x.ok.some(m => need(m) > 0.3)).sort((a, b) => b.score - a.score);
  const restTxt = avoid.size ? (avoid.size > 4 ? `; resting ${avoid.size} muscle groups trained in the last 48 h` : `; resting ${[...avoid].map(bmMuscleName).join(", ").toLowerCase()} (trained in the last 48 h)`) : "";
  if (!groups.length) return {focus: "Recovery / easy cardio", exercises: ["bike"], muscles: [], avoid: [...avoid], reason: "Nothing is clearly under target that hasn't just been trained" + restTxt + "."};
  const G = groups[0], main = G.ok.slice(0, 2);
  const extra = Object.keys(T).filter(m => !avoid.has(m) && !BM_GROUPS[G.g].includes(m) && m !== "forearms" && need(m) > 0.5).sort((a, b) => need(b) - need(a))[0];
  const ex = [], take = (m, n) => { for (const id of bmExercisesFor(m, 8, done, avoid)) { if (n <= 0) break; if (!ex.includes(id)) { ex.push(id); n--; } } };
  main.forEach(m => take(m, 1)); if (main.length < 2) take(main[0], 1);
  if (extra) take(extra, 1);
  const muscles = [...main, ...(extra ? [extra] : [])];
  return {focus: G.g + (extra ? " + " + bmMuscleName(extra).toLowerCase() : ""), exercises: ex, muscles, avoid: [...avoid],
    reason: `${muscles.map(m => `${bmMuscleName(m)} ${bmNum(load[m] || 0)}/${T[m]}`).join(", ")} sets this week${restTxt}.`};
}
