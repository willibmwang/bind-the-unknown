"""The panel machinery section 1.2 and section 4 render through.

This module holds no knob and no case. It is the dispatch wrapper, the replay reader and
the renderers, which are the same on every run and are the part of the notebook a reader
scrolls past rather than reads. The notebook keeps what a presenter changes: `LIVE`, the
per-tool `MODE`, the wall budget, the poll ceiling and which accession each panel asks
about. Lifting this out took cell 7 from 663 lines to roughly 150, so the notebook page
opens on the calls and their output instead of on their plumbing.

Nothing here was rewritten in the move. Every function is the body that ran in the cell,
and the one structural change is that the notebook state these functions used to read as
cell globals now arrives through `configure`, which the notebook calls once before the
first panel.
"""
import hashlib, json, textwrap, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from hpc_pilots import chief
from hpc_pilots.tools import TOOLS, tool

try:
    import py3Dmol                     # pinned in notebooks/requirements.txt
except ImportError:                    # the static figure below does not need it
    py3Dmol = None

# ----------------------------------------------------------------- what the notebook sets
# `configure` is the whole seam between this module and the notebook. These names are the
# cell globals the renderers used to close over, and they are rebound once per run rather
# than threaded through every signature, which would have rewritten seventeen bodies to
# move them.
ROOT = None
RUNS = None
FOLDED_JSON = None
FOLDED_PDB = None
RECORDED = {}
LIVE = True
MODE = {}
WALL_BUDGET_SECONDS = 600


def configure(*, root, runs, folded_json, folded_pdb, live, mode, wall_budget_seconds):
    """Bind this module to one run of the notebook, and report what it found on disk.

    Returns the two counts cell 7 prints, so the notebook states the registry size and the
    recording count from what this module actually resolved rather than from a second walk
    of the same directory that could disagree with it.
    """
    global ROOT, RUNS, FOLDED_JSON, FOLDED_PDB, RECORDED, LIVE, MODE, WALL_BUDGET_SECONDS
    ROOT, RUNS = root, runs
    FOLDED_JSON, FOLDED_PDB = folded_json, folded_pdb
    LIVE, MODE, WALL_BUDGET_SECONDS = live, dict(mode), wall_budget_seconds
    # Read off the corpus directory rather than listed by hand, so recording one more tool
    # makes it replayable here without editing the notebook, and so the count printed at the
    # end cannot drift from what is actually on disk. `hpc_pilots.exercise` names each file
    # after its tool with the dot replaced by an underscore, so the registry is matched
    # against that spelling instead of the filename being parsed.
    RECORDED = {name: RUNS / "toolcalls" / f"{name.replace('.', '_')}.json"
                for name in TOOLS
                if (RUNS / "toolcalls" / f"{name.replace('.', '_')}.json").exists()}
    return {"tools": len(TOOLS), "recorded": len(RECORDED),
            "reaches_network": sum(1 for s in TOOLS.values() if s.reaches_network),
            "py3Dmol": py3Dmol is not None}


# The four confidence bands AlphaFold DB established, which is the vocabulary anyone who reads
# a predicted structure already has. The hues keep AlphaFold's order, warm meaning do not
# trust, with the steps pushed apart in lightness: the literal AlphaFold palette collapses
# under deuteranopia on its orange and yellow pair, OKLab dE 6.5 against a floor of 8, and
# these clear it at 13.3. Computed with a Vienot dichromat projection, not eyeballed. Every
# band is named in words in the legend too, so the reading never rests on colour alone.
PLDDT_BANDS = ((90, 100.01, "#133C8C", "very high, 90 and above"),
               (70, 90, "#5BA8D8", "confident, 70 to 90"),
               (50, 70, "#F0C020", "low, 50 to 70"),
               (0, 50, "#D4551E", "very low, under 50"))
INK, INK2, SURFACE = "#1A1A1A", "#5A5A5A", "#FFFFFF"

# ----------------------------------------------------------------- dispatching for real
def declared_of(name):
    """What the registry advertises for this tool, kept in its own key for the whole section.

    `declared` and `measured` are two different claims and a panel that printed one where the
    other belongs would make every cost claim in the repository unfalsifiable.

    The record is read straight out of `TOOLS` rather than through `tools.tool`, which also
    checks that the tool's declared imports are present and raises when they are not. A replay
    needs the declaration and not the backend: `intake.rank_targets` declares `Bio`, which this
    kernel does not have, so `tool` raises for it before any panel could print.
    """
    spec = TOOLS[name]
    return {"cost_seconds": spec.cost_seconds, "cost_network_calls": spec.cost_network_calls,
            "worst_case_seconds": spec.worst_case_seconds,
            "reaches_network": spec.reaches_network,
            "missing_dependencies": spec.available()}


def dispatch(name, arguments=None, wall_budget_seconds=WALL_BUDGET_SECONDS):
    """Call one registered tool for real, and return the envelope beside what it spent.

    Nothing here reimplements dispatch: `chief.call` runs the policy gate, the semantic gate,
    the wall-budget gate and the tool itself. The wrapper adds the two things a panel needs
    that `chief.call` deliberately does not supply.

    `chief.schema_conformance` is the first. `chief.call` does not run it, by design, because a
    violation of the schema the tool emitted means the decoder ignored the grammar rather than
    the policy making a choice, so it raises instead of being charged. Four kinds of refusal
    are invisible without it, from an argument the schema required and the reply left out to
    one the tool never declared, and for a panel a refusal is something to render and not an
    exception to hide, so it is caught here and returned.

    An outer clock and an outer request count are the second. The envelope leaves `seconds` and
    `network_calls` as None for anything refused before dispatch, which is right, since a
    refusal is not a measurement of work. A panel still has to print something, so the outer
    window goes in its own key and is never written into the envelope's fields.
    """
    arguments = dict(arguments or {})
    out = {"tool": name, "arguments": arguments, "mode": "live", "envelope": None,
           "refused": None, "declared": declared_of(name), "measured": None,
           "fallback": None, "absent": None}
    started = time.perf_counter()
    with chief.counting_requests() as tally:
        try:
            # Inside the window, because `tool` is also where a declared dependency this
            # kernel lacks surfaces, and that is a panel to render rather than a traceback.
            spec = tool(name)
            chief.schema_conformance(spec, arguments)
            out["envelope"] = chief.call(spec, arguments, context=chief.context(
                None, now=0, wall_budget_seconds=wall_budget_seconds))
        except chief.ToolCallRejected as refused:
            out["refused"] = {"exception": type(refused).__name__, "message": str(refused)}
        except Exception as broke:
            # chief.call turns an endpoint failure into an envelope rather than an exception,
            # so this path is for the unanticipated one. A panel that tracebacks in front of an
            # audience is worse than a panel that reports what broke.
            out["absent"] = f"{type(broke).__name__}: {broke}"
    # The outer window counts what the inner one did: a nested request window adds its total
    # back into the enclosing one on exit, so outer and envelope agree on every call that ran.
    out["measured"] = {"seconds": time.perf_counter() - started,
                       "network_calls": tally["made"]}
    return out


# `record` says whether a tool writes its own artifact and changes nothing it asks of an
# endpoint, so a panel that passes record=False matches a recorded cell that left the default
# in place. Nothing else is normalised away: every other argument is part of the question.
NOT_ASKED_OF_THE_ENDPOINT = ("record",)


def _asked(arguments):
    return json.dumps({k: v for k, v in arguments.items()
                       if k not in NOT_ASKED_OF_THE_ENDPOINT}, sort_keys=True)


def replay(name, arguments=None):
    """The same keys, rebuilt from a committed recording. Touches nothing.

    Two recorded shapes, and they carry different amounts. `runs/folded_structure/` holds a
    whole tool return, payload included, which is why the fold panel replays in full. A
    `runs/toolcalls/` trace row holds the arguments, the status, the measured seconds and the
    request count, and no payload at all, so a replayed retrieval panel can show that the call
    happened and what it spent and cannot show the rows that came back. Fields the recording
    does not carry are left None and print as absent rather than being filled in.
    """
    arguments = dict(arguments or {})
    out = {"tool": name, "arguments": arguments, "mode": "replay", "envelope": None,
           "refused": None, "declared": declared_of(name), "measured": None,
           "fallback": None, "absent": None}

    if name == "structures.folded":
        if not FOLDED_JSON.exists():
            out["absent"] = (f"{FOLDED_JSON.relative_to(ROOT)} is not on this machine, "
                             "so there is no recorded fold to replay")
            return out
        p = json.loads(FOLDED_JSON.read_text())
        out["envelope"] = {
            "tool": name, "tool_version": None, "arguments": arguments,
            "status": "ok" if p["folded"] else "empty", "value": p, "rows": None,
            "citable": None, "verified": None, "source": p["endpoint"], "source_kind": None,
            "available_from": p["model_date"], "note": "", "detail": None,
            # The record's own measurement, reprinted. `seconds` here is a warm cache hit and
            # not the cost of folding something new, which the line under the cost table says
            # in the words of the module that measured it.
            "seconds": p["seconds"], "network_calls": p["network_calls"]}
        out["measured"] = {"seconds": p["seconds"], "network_calls": p["network_calls"]}
        prov = p.get("provenance", {})
        out["recording"] = {"path": str(FOLDED_JSON.relative_to(ROOT)), "calls": 1,
                            "by_status": {out["envelope"]["status"]: 1}, "measured": None,
                            "recorded_at_utc": prov.get("recorded_at_utc", "?"),
                            "git_commit": prov.get("git_commit", "?")[:12],
                            "git_dirty": prov.get("git_dirty")}
        return out

    path = RECORDED.get(name)
    if path is None or not path.exists():
        out["absent"] = (f"no recording for {name} on this machine; runs/toolcalls/ holds "
                         f"{len(RECORDED)} of the {len(TOOLS)} registered tools "
                         f"({', '.join(sorted(RECORDED))})")
        return out
    rec = json.loads(path.read_text())
    want = _asked(arguments)
    row = next((r for r in rec["trace"] if _asked(r["arguments"]) == want), None)
    if row is None:
        got = sorted({_asked(r["arguments"]) for r in rec["trace"]})
        out["absent"] = (f"{path.relative_to(ROOT)} records no call with these arguments; it "
                         f"holds {', '.join(got)}")
        return out
    out["envelope"] = {
        "tool": row["tool"], "tool_version": row["tool_version"],
        "arguments": row["arguments"], "status": row["tool_status"],
        # A trace row keeps the cost and the provenance and drops `value`, so there is no
        # payload to show. None, and the panel says which field is missing and why.
        "value": None, "rows": row["rows"], "citable": row["citable"],
        "verified": None, "source": None, "source_kind": row["source_kind"],
        "available_from": None, "note": row["result"], "detail": row["rejected"],
        "seconds": row["tool_seconds"], "network_calls": row["tool_network_calls"]}
    out["measured"] = {"seconds": row["tool_seconds"],
                       "network_calls": row["tool_network_calls"]}
    out["recording"] = {"path": str(path.relative_to(ROOT)), "calls": rec["calls"],
                        "by_status": rec["by_status"], "measured": rec["measured"],
                        "recorded_at_utc": rec["provenance"]["recorded_at_utc"],
                        "git_commit": rec["provenance"]["git_commit"][:12],
                        "git_dirty": rec["provenance"]["git_dirty"]}
    return out


def call_tool(name, arguments=None, wall_budget_seconds=WALL_BUDGET_SECONDS, force=None,
              recorded_as=None):
    """One entry point for both modes, so every panel below is the same object either way.

    `force` pins a panel to one mode and overrides `MODE`. The refusal panel is pinned live
    because the wall-budget gate runs locally and reaches no network, so it is the same call in
    a room with no connection.

    `recorded_as` names the recorded cell a replay should show when the live arguments are not
    one the corpus holds. `similarity.homologs` goes out under a 20 s poll ceiling, which is
    not an argument the committed grid carries, so its replay shows the grid's own cell on the
    same query and prints those arguments rather than the live ones.
    """
    mode = force or MODE.get(name) or ("live" if LIVE else "replay")
    recorded = dict(recorded_as) if recorded_as is not None else arguments
    if mode == "replay":
        return replay(name, recorded)
    out = dispatch(name, arguments, wall_budget_seconds)
    env = out["envelope"]
    # An outage is a status, not an exception: structures.folded comes back `empty` with a
    # reason code and structures.predicted comes back `failed`. Either way the panel still
    # renders, and where a recording exists the section falls back to it rather than going
    # blank, with the fallback named on screen and the failed attempt's own cost beside it,
    # because what a doomed call spent is still a measurement and the recording's cost is not
    # it: with the network unreachable the fold attempt spends about 3 s and 2 requests.
    # `unavailable` is deliberately not in that list. It is what the wall-budget gate returns
    # when a tool declines itself, which is a result and not an outage, and falling back from
    # it would hide the one refusal this section exists to show.
    if out["absent"] or (env is not None and env["status"] in ("empty", "failed")):
        why = out["absent"] or f"the live call came back {env['status']}"
        back = replay(name, recorded)
        if back["envelope"] is not None:
            back["fallback"] = f"{why}, so this panel is the committed recording instead"
            back["live_attempt"] = out["measured"]
            return back
        out["fallback"] = f"{why}, and no recording exists to fall back to"
    return out


# ----------------------------------------------------------------- rendering one call
def _short(value, width=64):
    """Long strings print as a length and a digest. A 306-residue sequence is an argument a
    panel has to show it passed, and printing it in full buries everything around it. The
    width is 64 so a sha256 prints whole rather than being digested again."""
    if isinstance(value, str) and len(value) > width:
        return f"<{len(value)} chars, sha256 {hashlib.sha256(value.encode()).hexdigest()[:12]}>"
    return repr(value)


def _scalars(payload, lead=(), omit=(), limit=8):
    """Scalar payload fields, the named ones first. Lists and dicts are counted, not dumped:
    this payload carries a 306-entry pLDDT vector and the whole PDB as a string.

    `omit` drops a field from this list for a panel that prints it in full below instead. A
    standing warning and a cyclisation chemistry are both long enough to come out as a digest
    here, which is the one shape a caveat must never take."""
    if not isinstance(payload, dict):
        return []
    scalar = lambda v: isinstance(v, (int, float, bool, str, type(None)))
    # `lead` is filtered by the same test as everything else. A named key is a request for an
    # order, not a licence to dump: `literature_key` is a dict carrying two dozen family names
    # and thirty PMIDs, and one of those in a panel buries every line around it.
    keys = [k for k in lead if k in payload and scalar(payload[k]) and k not in omit]
    keys += [k for k, v in payload.items()
             if k not in keys and k not in omit and scalar(v)]
    return [(k, _short(payload[k]) if isinstance(payload[k], str) else repr(payload[k]))
            for k in keys[:limit]]


def panel(result, lead=(), omit=()):
    """Print one tool call: live or replayed, ran or refused, through one code path.

    The mode is in the header of every panel and not only in the knob, because an audience
    must not be able to mistake a replay for a live call, or a call that never left this
    machine for one that reached an endpoint. The refusal panel dispatches for real and still
    makes no request, because the gate that refuses it runs here, so the tag is computed from
    the measured request count rather than from the mode alone.
    """
    env, ref, m = result["envelope"], result["refused"], result["measured"]
    if result["mode"] == "replay":
        tag = "replay of a committed recording, no network touched"
    elif m and m["network_calls"]:
        tag = f"LIVE, dispatched for real just now, {m['network_calls']} request(s) out"
    else:
        tag = "LIVE, dispatched for real just now, and no request left this machine"
    head = result["tool"]
    if env is not None and env.get("tool_version") is not None:
        head += f" v{env['tool_version']}"
    print(f"{head}    [{tag}]")
    if result.get("fallback"):
        print(textwrap.fill(result["fallback"], 93, initial_indent="  fallback       ",
                            subsequent_indent="                 "))
    if result.get("live_attempt"):
        a = result["live_attempt"]
        print(f"                 the attempt itself spent {a['seconds']:.3f} s and tried "
              f"{a['network_calls']} request(s) before falling back")
    print("  arguments      " +
          (", ".join(f"{k}={_short(v)}" for k, v in result["arguments"].items()) or "none"))

    if env is None and ref is None:                       # nothing on disk and nothing live
        print(textwrap.fill(result["absent"] or "no call and no recording", 93,
                            initial_indent="  not shown      ",
                            subsequent_indent="                 "))
        print()
        return result

    d = result["declared"]
    if env is None:                                       # raised by schema_conformance
        print(f"  refused        {ref['exception']}, before any gate charged anything")
        print(textwrap.fill(ref["message"], 91, initial_indent="    ",
                            subsequent_indent="    "))
    else:
        # A field the recording does not carry prints as absent and never as a value, because
        # `citable False` and `citable not recorded` are different claims about the call.
        absent = lambda v: "not recorded" if (v is None and result["mode"] == "replay") else v
        print(f"  status         {env['status']}"
              f"    citable {absent(env['citable'])}    verified {absent(env['verified'])}"
              f"    source_kind {absent(env['source_kind'])}")
        if env["note"]:
            print(textwrap.fill(str(env["note"]), 91, initial_indent="  note           ",
                                subsequent_indent="                 "))
        if env["source"]:
            print(f"  source         {env['source']}")
        if env["available_from"]:
            print(f"  available from {env['available_from']}")
        if env["seconds"] is None:
            print("  measured       the envelope records None and not zero: the call was "
                  "refused before dispatch,")
            print("                 so there is no work to measure. "
                  f"The gate itself took {m['seconds'] * 1000:.2f} ms "
                  f"and {m['network_calls']} requests.")
        else:
            print(f"  measured       {env['seconds']:.3f} s, {env['network_calls']} outbound "
                  f"request(s)" + ("" if result["mode"] == "live" else
                                   ", as the recording measured them"))
    dn = "none declared" if d["cost_network_calls"] is None else f"{d['cost_network_calls']}"
    wc = ("none declared" if d["worst_case_seconds"] is None
          else f"{d['worst_case_seconds']:.0f} s")
    # The 0.0 note is for a tool that reaches an endpoint and declares no cost for it. On
    # `chemistry.library_spec` the same 0.0 is the real number, because the tool reads local
    # tables and declares no request, so the note would be false exactly where it is loudest.
    unmeasured = not d["cost_seconds"] and d["reaches_network"]
    print(f"  declared       {d['cost_seconds']} s, {dn} request(s), worst case {wc}"
          f"{'   (0.0 means unmeasured, not free)' if unmeasured else ''}")
    if env is not None and env["value"] is not None:
        for k, v in _scalars(env["value"], lead, omit):
            print(f"      {k:<24} {v}")
    elif env is not None and result["mode"] == "replay":
        print("  payload        not recorded. A runs/toolcalls/ trace row keeps the "
              "arguments, the status and")
        print("                 the cost and drops the tool's return, so a replay here can "
              "show that the call")
        print("                 happened and what it spent, and cannot show what came back.")
    if "recording" in result:
        r = result["recording"]
        print(f"  recording      {r['path']}, {r['calls']} calls {r['by_status']}, "
              f"{r['recorded_at_utc'][:19]}Z at {r['git_commit']}"
              f"{' (dirty tree)' if r['git_dirty'] else ''}")
    print()
    return result


def rows_table(result, rows_of, columns, limit=3, caption=""):
    """A few of the rows a call actually returned, printed under its panel.

    The structural lanes hand back coordinates and this section draws them. The retrieval
    lanes hand back text and tables, and a status line alone hides the thing they are for, so
    three rows print whole: a homolog with its identity and E-value, a family with its residue
    range, a paper with its date. `rows_of` takes the payload and returns an iterable of rows,
    and each column is a label, a width and a function of one row. Anything that raises on the
    way prints as a question mark, because one odd field is not worth losing the table over.
    """
    env = result["envelope"]
    if env is None or env["value"] is None:
        return result
    try:
        rows = list(rows_of(env["value"]))[:limit]
    except Exception as broke:
        print(f"      rows not shown: {type(broke).__name__}: {broke}\n")
        return result
    if not rows:
        print("      no rows, which for a novel target is the ordinary answer and a recorded "
              "absence\n")
        return result
    if caption:
        print(f"      {caption}")
    print("      " + "  ".join(f"{label:<{width}}" for label, width, _ in columns))
    for row in rows:
        cells = []
        for _, width, read in columns:
            try:
                text = str(read(row))
            except Exception:
                text = "?"
            cells.append(f"{text[:width]:<{width}}")
        print("      " + "  ".join(cells))
    print()
    return result


def group(number, title, tools_and_endpoints):
    """One rule and one heading per group, so ten panels read as a pipeline and not a list.

    The grouping is by the question a call answers, which is the order the project itself
    works in: candidates, then what is known, then what the structure looks like, then what
    could bind it.
    """
    print("\n" + "─" * 93)
    print(f"{number} · {title.upper()}" if number else title.upper())
    print(textwrap.fill(tools_and_endpoints, 93))
    print("─" * 93)


# ----------------------------------------------------------------- reading the structure
def ca_trace(pdb_text):
    """Residue numbers, CA coordinates and per-residue confidence, by fixed PDB column.

    A direct column read rather than biopython, because it is the faster of the two on a file
    this size and it does not mind that the ESMFold file carries neither a TER nor an END
    record. The two agree exactly on coordinates and B-factors, checked.
    """
    resnum, xyz, bfac = [], [], []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in (" ", "A"):
            resnum.append(int(line[22:26]))
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            bfac.append(float(line[60:66]))
    return np.array(resnum), np.array(xyz), np.array(bfac)


def plddt_on_0_100(bfac):
    """pLDDT from the B-factor column, normalised to the 0 to 100 convention the bands use.

    The scale is detected and not assumed. This endpoint writes pLDDT into B-factor on a 0 to 1
    scale, while the tool's own `confident_plddt` parameter defaults to 70.0 and AlphaFold's
    published bands run 0 to 100. Reading the column as it stands would colour every residue in
    the lowest band and look like a catastrophic prediction rather than a unit error. AlphaFold
    DB's own PDB files already come 0 to 100, so a detector is what lets one parser serve both
    lanes.
    """
    if bfac.size and bfac.max() <= 1.0:
        return bfac * 100.0, "0 to 1 in the file, multiplied by 100 to match the bands"
    return bfac.copy(), "already 0 to 100 in the file, left as it is"


def band_colour(v):
    for lo, hi, colour, _ in PLDDT_BANDS:
        if lo <= v < hi:
            return colour
    return PLDDT_BANDS[0][2]


def structure_viewer(pdb_text, resnum, plddt, width=460, height=460):
    """A py3Dmol cartoon the viewer can spin, coloured by confidence band.

    Bands are applied by residue number rather than by selecting on the B-factor column,
    because the residue lists are computed here from the scale this notebook normalised and so
    the colouring cannot disagree with the static figure beside it.

    py3Dmol pulls 3Dmol.js from a CDN when the cell renders, so this view needs a browser with
    a network connection, and GitHub strips scripts out of notebook output, which leaves it
    blank for anyone reading the notebook there. The matplotlib figure is what survives both.
    """
    if py3Dmol is None:
        print("py3Dmol is not installed, so the interactive view is unavailable and the "
              "static figure above is the whole of what this panel can show. "
              "notebooks/requirements.txt pins 2.5.5.")
        return None
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "#BBBBBB"}})
    for lo, hi, colour, _ in PLDDT_BANDS:
        resi = [int(r) for r, v in zip(resnum, plddt) if lo <= v < hi]
        if resi:
            view.setStyle({"resi": resi}, {"cartoon": {"color": colour}})
    view.zoomTo()
    return view


def structure_figure(resnum, xyz, plddt, title, subtitle, confident_plddt=70.0, runs=()):
    """The same structure as a static figure, which is the one that survives GitHub.

    The left panel is a hand-rolled orthographic projection onto the first two principal axes,
    drawn back to front with a surface-coloured halo under each segment. mplot3d is not used
    because it will not depth-sort a line collection, so a 306-residue fold does not occlude
    itself and reads as flat wire. Orthographic also makes the scale bar a true distance.
    """
    fig = plt.figure(figsize=(13.6, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.1,
                          left=0.02, right=0.97, top=0.78, bottom=0.17)

    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    centred = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    for i in range(2):                      # pin the SVD sign so the view reproduces
        if vt[i, i] < 0:
            vt[i] *= -1
    proj = centred @ vt.T                   # long axis, second axis, depth
    segs = np.stack([proj[:-1, :2], proj[1:, :2]], axis=1)
    depth = (proj[:-1, 2] + proj[1:, 2]) / 2
    order = np.argsort(depth)
    near = (depth[order] - depth.min()) / max(np.ptp(depth), 1e-9)
    cols = np.array([band_colour(v) for v in plddt[:-1]])[order]
    ax.add_collection(LineCollection(segs[order], colors=SURFACE, linewidths=8.0,
                                     capstyle="round", zorder=2))
    ax.add_collection(LineCollection(segs[order], colors=cols, linewidths=2.2 + 2.6 * near,
                                     capstyle="round", zorder=3))
    for i, lab in ((0, "N"), (len(proj) - 1, "C")):
        ax.scatter(*proj[i, :2], s=150, facecolor=SURFACE, edgecolor=INK, lw=1.5, zorder=4)
        ax.text(*proj[i, :2], lab, ha="center", va="center", fontsize=10, fontweight="bold",
                color=INK, zorder=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    x0, x1 = proj[:, 0].min() - 4, proj[:, 0].max() + 4
    y0, y1 = proj[:, 1].min() - 4, proj[:, 1].max() + 4
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.plot([x1 - 22, x1 - 2], [y0 + 3, y0 + 3], color=INK2, lw=2.2, solid_capstyle="butt")
    ax.text(x1 - 12, y0 + 4.4, "20 Å", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_title(f"backbone, {len(resnum)} Cα, orthographic on its two principal axes",
                 fontsize=11.5, color=INK, pad=10, loc="left")
    # Under the panel rather than inside it: the fold fills the frame, and a legend laid over
    # the chain hides the thing the panel is for.
    ax.legend(handles=[Line2D([], [], color=c, lw=5, label=l) for _, _, c, l in PLDDT_BANDS],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False,
              fontsize=9.5, labelcolor=INK, handlelength=1.4, borderpad=0.1,
              labelspacing=0.5, columnspacing=1.8)

    ax2 = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    for lo, hi, colour, _ in PLDDT_BANDS:
        ax2.axhspan(lo, min(hi, 100), color=colour, alpha=0.10, lw=0)
    ax2.plot(resnum, plddt, color=INK, lw=1.2, zorder=3)
    ax2.scatter(resnum, plddt, c=[band_colour(v) for v in plddt], s=9, zorder=4,
                edgecolor="none")
    ax2.axhline(confident_plddt, color=INK2, lw=1.1, ls=(0, (5, 3)), zorder=5)
    ax2.text(resnum[-1] - 3, confident_plddt + 2, f"threshold {confident_plddt:.0f}",
             ha="right", va="bottom", fontsize=9, color=INK2)
    ax2.axhline(plddt.mean(), color=INK, lw=1.0, zorder=5)
    # Left of the threshold label and below its own line, so the two never collide: they sit
    # within eight pLDDT units of each other on this structure.
    ax2.text(resnum[0] + 6, plddt.mean() - 2.5, f"mean {plddt.mean():.1f}", ha="left",
             va="top", fontsize=9, color=INK)
    # The stretches that never clear the threshold are the useful reading of this panel, far
    # more than its mean: they are where the model is telling you not to believe it.
    for a, b in runs:
        ax2.plot([a, b], [4, 4], color=INK2, lw=3.0, solid_capstyle="butt", zorder=6)
        if b - a > 12:
            ax2.text((a + b) / 2, 7.5, f"{a}–{b}", ha="center", va="bottom", fontsize=8.5,
                     color=INK2)
    ax2.set_xlim(resnum[0], resnum[-1])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("confidence", fontsize=10, color=INK2)
    ax2.set_xlabel("residue position in the folded sequence", fontsize=10, color=INK2)
    ax2.set_title("per-residue confidence, and the stretches that never clear the threshold",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color("#C8C8C8")
    ax2.tick_params(colors=INK2, labelsize=9)

    fig.suptitle(title, x=0.02, y=0.955, ha="left", fontsize=15.5, color=INK,
                 fontweight="bold")
    fig.text(0.02, 0.855, subtitle, ha="left", fontsize=9.5, color=INK2)
    return fig


def folded_sequence(pdb_text):
    """The one-letter sequence of the chain in a folded PDB, recovered from the file itself.

    Retyping a 306-residue sequence into the notebook would break the rule that nothing here
    is retyped, and the demo cell checks what this returns against the record's own
    `sequence_sha256`, so a drifted file is a visible mismatch and not a different live call.
    """
    # The twenty-letter mapping is written out rather than imported from biopython, which is
    # not installed in every kernel this notebook is opened under and would be a dependency
    # carried for one lookup table. An unrecognised residue becomes "X" rather than raising,
    # so a file with a modified residue still yields a sequence and a visible mismatch.
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
        "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    seen, seq = set(), []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in (" ", "A"):
            key = int(line[22:26])
            if key not in seen:
                seen.add(key)
                seq.append(three_to_one.get(line[17:20].strip(), "X"))
    return "".join(seq)


# ----------------------------------------------------------------- the plan, before it runs
def recorded_cost(name):
    """The median this tool's committed recording measured, or None where it has no recording.

    Read out of the artifact every time rather than copied into this cell, so re-recording a
    tool moves the plan table with it and the two cannot disagree.
    """
    path = RECORDED.get(name)
    if path is None or not path.exists():
        return None
    rec = json.loads(path.read_text())
    m = rec.get("measured") or {}
    return {"seconds": m.get("seconds_median"), "requests": m.get("network_calls_median"),
            "calls": rec.get("calls")}


# ----------------------------------------------------------------- the plan, before it runs
def plan_table():
    """The mode and cost table cell 7 prints before any panel runs.

    It lives here because it reads `declared_of` and `recorded_cost`, which are internal to
    this module, and because it is a fixed report rather than part of the case.
    """
    print(f"LIVE = {LIVE}  ->  "
          + ("panels dispatch for real, except where MODE pins one to a replay"
             if LIVE else "panels replay committed recordings and touch no network"))
    print(f"registry: {len(TOOLS)} tools, "
          f"{sum(1 for s in TOOLS.values() if s.reaches_network)} of them reach the network, "
          f"{len(RECORDED)} with a committed recording")
    print(f"py3Dmol  : {'present' if py3Dmol else 'absent'}"
          "   (the spinnable view; the static figure beside it does not need it)")

    print(f"\n  {'tool':<27}{'mode':<17}{'declared':<24}recorded median")
    for name in sorted(TOOLS):
        pinned = MODE.get(name)
        mode = pinned or ("live" if LIVE else "replay")
        d, c = declared_of(name), recorded_cost(name)
        dn = "?" if d["cost_network_calls"] is None else f"{d['cost_network_calls']} req"
        shown = f"{d['cost_seconds']:.1f} s, {dn}"
        if c is None:
            got = "no recording"
        elif c["seconds"] is None:
            got = f"{c['calls']} calls, no timed median"
        else:
            got = f"{c['seconds']:.3f} s, {c['requests']} req, {c['calls']} calls"
        print(f"  {name:<27}{mode + (' (pinned)' if pinned else ''):<17}{shown:<24}{got}")
        if d["missing_dependencies"]:
            print(f"  {'':<27}needs {', '.join(d['missing_dependencies'])}, "
                  "absent in this kernel")



# ----------------------------------------------------------------- the workflow flowchart
# Three hues, and nothing is identified by hue alone: every node carries its own words and
# every edge its own label, so the diagram survives being printed in grey. The trio was
# checked rather than chosen -- OKLab dE under deuteran, protan and tritan simulation, worst
# adjacent pair 9.2, normal-vision floor 24.0, all three inside the lightness band and over
# the chroma floor -- which is the discipline `PLDDT_BANDS` above is already under.
ARM1, ARM2, BREAK = "#2a78d6", "#1baf7a", "#eb6834"
PALE1, PALE2, PALEB = "#eaf2fc", "#e7f7f0", "#fdeee6"


class _Flow:
    """A top-down cursor that computes every height from the text it is given.

    Laying a flowchart out by hand in axes fractions is how the first version of this
    collided its own labels: a box whose height is guessed does not grow when its body gains
    a line, and every number in this figure comes out of a record that is free to change. So
    nothing here is positioned absolutely. A node's height is its line count times the line
    height for its font size, an edge's height is its label's line count, and the cursor walks
    down. Re-running with a longer label moves the figure rather than breaking it.
    """

    def __init__(self, ax, axes_height_inches, left, width, gutter, top):
        self.ax, self.h_in = ax, axes_height_inches
        self.left, self.width, self.gutter = left, width, gutter
        self.y = top

    def line(self, size, spacing=1.45):
        """One text line, as a fraction of the axes height."""
        return size * spacing / 72.0 / self.h_in

    def node(self, title, body, edge, face, x, width, title_size=10.5, body_size=9.0,
             pad=0.007):
        from matplotlib.patches import FancyBboxPatch

        lines = body.count("\n") + 1 if body else 0
        th, bh = self.line(title_size, 1.3), self.line(body_size)
        height = pad * 2 + th + (0.004 + lines * bh if lines else 0)
        bottom = self.y - height
        self.ax.add_patch(FancyBboxPatch(
            (x, bottom), width, height,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            linewidth=1.4, edgecolor=edge, facecolor=face, zorder=2))
        self.ax.text(x + 0.012, self.y - pad, title, fontsize=title_size, color=edge,
                     fontweight="bold", va="top", ha="left", zorder=3)
        if body:
            self.ax.text(x + 0.012, self.y - pad - th - 0.004, body, fontsize=body_size,
                         color=INK, va="top", ha="left", linespacing=1.45, zorder=3)
        # The cursor moves by the height this node actually took. Two callers used to
        # subtract a height they recomputed by hand from the same font sizes, which is a
        # second copy of this arithmetic that a longer label silently invalidates.
        self.y = bottom
        return bottom, height

    def row(self, title, body, out, edge, face, out_face, **kw):
        """A stage and its output, side by side, with the arrow between them."""
        top = self.y
        b1, h1 = self.node(title, body, edge, face, self.left, self.width, **kw)
        self.y = top                      # the two nodes in a row start at the same height
        b2, h2 = self.node("out", out, edge, out_face, self.left + self.width + self.gutter,
                           self.width, title_size=8.5, **kw)
        mid = top - min(h1, h2) / 2
        self.ax.annotate("", xy=(self.left + self.width + self.gutter, mid),
                         xytext=(self.left + self.width, mid),
                         arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
                                         linewidth=1.3, color=edge, shrinkA=2, shrinkB=2),
                         zorder=1)
        self.y = top - max(h1, h2)
        return self.y

    def edge(self, label, colour=INK2, dashed=False, size=8.5, pad=0.010):
        """A labelled vertical arrow. The gap is as tall as the label needs."""
        lines = label.count("\n") + 1 if label else 0
        height = pad * 2 + lines * self.line(size)
        x = self.left + self.width / 2
        self.ax.annotate("", xy=(x, self.y - height), xytext=(x, self.y),
                         arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                         linewidth=1.6, color=colour, shrinkA=0, shrinkB=0,
                                         linestyle=(0, (4, 3)) if dashed else "solid"),
                         zorder=1)
        if label:
            self.ax.text(x + 0.014, self.y - height / 2, label, fontsize=size, color=colour,
                         va="center", ha="left", linespacing=1.45, style="italic", zorder=3)
        self.y -= height
        return self.y


def _band(ax, y_top, y_bottom, title, subtitle, colour, face, right=1.0):
    """The arm label in the left margin, with its rows tinted behind them."""
    height = y_top - y_bottom
    ax.add_patch(plt.Rectangle((-0.035, y_bottom), right + 0.035, height, facecolor=face,
                               edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((-0.035, y_bottom), 0.012, height, facecolor=colour,
                               edgecolor="none", zorder=1))
    ax.text(-0.052, y_top - 0.012, f"{title}\n{subtitle}", fontsize=10.5, color=colour,
            fontweight="bold", va="top", ha="right", linespacing=1.7, zorder=3)


def workflow_figure(chain, case=0, gain_shared="+0.173 to +0.321",
                    gain_unshared="+0.010 to +0.033", height=14.6):
    """The two arms as a flowchart: what each stage takes, what it returns, what it hands on.

    Every count is read out of the chain record passed in, so the picture cannot drift from
    the run. The three things it is drawn to make visible, which a list of tools does not:

    * the single bottleneck -- one stage supplies all four facts the other nine tools wait on;
    * the stage that used to cover less than its input, now exhaustive by construction;
    * the two arrows that are *not* there, drawn dashed in a third hue rather than omitted.

    `gain_shared` and `gain_unshared` come from the per-batch summaries section 2.1 reads, and
    are passed in so the figure and the table quote the same source.
    """
    # The record holds one entry per reference the genome was identified against, because
    # which reference is used changes what every later stage receives. `case` picks which one
    # the figure draws, and the subtitle names it, so a reader is never looking at a coverage
    # number without the setting that produced it.
    run = chain["cases"][case] if "cases" in chain else chain
    # Keyed on the tool each stage ran, not on its position. The pipeline's order is a design
    # decision and it has already changed once -- folding moved ahead of the surface stage so
    # that a chain with no deposited structure has coordinates to measure -- and a figure that
    # reads `stages[3]` draws whichever stage happens to sit third under the previous one's
    # caption. Keying on the tool also keeps this readable against records written under either
    # numbering, which matters because every record on disk predates the swap.
    by_tool = {s["tool"]: s for s in run["stages"]}
    st1 = by_tool["intake.rank_targets"]
    st2 = by_tool["intake.decompose + intake.rank_score"]
    st_surface = by_tool["geometry.surface_markers"]
    st_fold = by_tool["structures.folded"]
    st_lib = by_tool["chemistry.library_spec"]
    out = {1: st1["output"], 2: st2["output"]}
    fold_out, lib = st_fold["output"], st_lib["output"]
    # The stage's own in and out in one dict, for the surface row below.
    surf = {**st_surface["output"], **st_surface["input"]}
    # Which chains the fold stage covered. `chains_folded` exists only in records written after
    # the stage began folding every chain inside the ceiling; before that it folded one, so the
    # older records answer this question with a single target name.
    folded_count = fold_out.get("chains_folded")
    if folded_count is None:
        folded_count = 1 if fold_out.get("folded_target") else 0

    fig = plt.figure(figsize=(13.6, height), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes([0.105, 0.008, 0.885, 0.942])
    ax.set_xlim(-0.30, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")
    flow = _Flow(ax, axes_height_inches=height * 0.942, left=0.0, width=0.47, gutter=0.06,
                 top=0.995)

    fig.text(0.105, 0.992, "One genome in, and what each stage hands the next",
             fontsize=16, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.105, 0.972,
             f"{run['genome_accession']}  ·  {run['case']}  ·  every count is read from "
             "runs/onset_chain.json, written by  python -m hpc_pilots chain",
             fontsize=9.5, color=INK2, ha="left", va="top")

    flow.node("INPUT  ·  one pathogen genome accession",
              f"{run['genome_accession']}   ·   {out[1]['genome_length_nt']:,} nt   ·   "
              f"{out[1]['segments']} segment\n"
              "Nothing else: no target name, no shortlist, no structure.",
              INK, "#f3f3f1", 0.0, 0.47 * 2 + 0.06)
    flow.edge("genome_accession")

    arm1_top = flow.y + 0.006
    flow.row("1 · intake.rank_targets",
             f"NCBI · UniProt   ·   {st1['seconds']} s\n"
             f"in:  the accession, and nothing else",
             f"{out[1]['orfs_found']} ORFs called in six frames, "
             f"{out[1]['orfs_ranked']} ranked\n"
             f"{out[1]['identified']} identified against "
             f"{out[1]['reference_entries']:,} reviewed entries\n"
             f"{out[1]['unidentified']} not identified, and reported as such",
             ARM1, SURFACE, PALE1)
    flow.edge("target_name · protein_accession · residue_ranges · protein_sequence\n"
              "— all four facts the other nine tools are waiting on, from this one call",
              colour=ARM1)
    flow.row("2 · intake.decompose  +  rank_score",
             f"UniProt   ·   {st2['seconds']} s\n"
             f"in:  {out[1]['identified']} identified candidates",
             f"{out[2]['chains_out']} mature chains, each with a score\n"
             "score = 3·conservation + 2·exposed + 1.5·essential\n"
             "          + 0.5·min(1, structures/50) + 0.8·druggable\n"
             "every term is a stated weight; nothing here is fitted",
             ARM1, SURFACE, PALE1)
    flow.edge(f"one accession and one residue range per chain — "
              f"{surf['candidates_in']} of {out[2]['chains_out']} go on,\n"
              f"and {fold_out['foldable']} of {out[2]['chains_out']} can also supply their own "
              f"sequence to stage {st_fold['stage']}", colour=ARM1)
    # The fold row and the surface row, drawn in the order the RECORD ran them and labelled
    # with the numbers the record gave them. The order is not hardcoded either way: a run from
    # before the swap measured surfaces third and folded fourth, and drawing that run in
    # today's order would caption it with a pipeline it did not execute. The edge between them
    # is the earlier stage's own `hands_to_next`, which is the field that says what it handed
    # on -- so a record whose fold stage says nothing downstream read it draws exactly that.
    def fold_row():
        flow.row(f"{st_fold['stage']} · structures.folded",
                 f"ESM Atlas   ·   {st_fold['seconds']} s\n"
                 f"in:  an amino-acid sequence per chain — from stage {st2['stage']}",
                 f"{fold_out['foldable']} of "
                 f"{fold_out['foldable'] + fold_out['not_foldable']} chains could supply "
                 f"their own sequence\nfolded {folded_count}, "
                 f"{fold_out['folded_target']} shown: {fold_out['residues']} residues\n"
                 f"mean pLDDT {fold_out['mean_plddt']}, "
                 f"{round(100 * (fold_out['at_or_above_70'] or 0), 1)}% at or above 70\n"
                 "— the predictor's confidence in itself, not accuracy",
                 ARM1, SURFACE, PALE1)

    def surface_row():
        predicted = surf.get("measured_on_predicted_coordinates")
        computed = (f"{predicted} measured on predicted coordinates instead, never pooled"
                    if predicted is not None else
                    "deposited structures only — this run predates the predicted lane")
        flow.row(f"{st_surface['stage']} · geometry.surface_markers",
                 f"PDBe · RCSB · UniProt   ·   {st_surface['seconds']} s\n"
                 f"in:  {surf['candidates_in']} chains as accession + range"
                 + (f", and {surf['predicted_coordinates_in']} predicted structures"
                    if surf.get("predicted_coordinates_in") else ""),
                 f"{surf['rows_out']} rows for {surf['candidates_in']} chains: "
                 f"{surf['measured']} measured on depositions, "
                 f"{surf['unmeasured']} with none\n{computed}\n"
                 f"gate:  {surf['gate_yes']} yes  ·  {surf['gate_borderline']} borderline  ·  "
                 f"{surf['gate_no']} no",
                 ARM1, SURFACE, PALE1)

    first, second = ((st_fold, fold_row), (st_surface, surface_row))
    if st_surface["stage"] < st_fold["stage"]:
        first, second = (st_surface, surface_row), (st_fold, fold_row)
    first[1]()
    flow.edge(textwrap.fill(first[0].get("hands_to_next") or "", 64), colour=ARM1)
    second[1]()
    flow.edge(textwrap.fill(second[0].get("hands_to_next") or "", 64), colour=ARM1)
    flow.edge("", colour=BREAK, dashed=True, pad=0.007)
    flow.node("THE ARROW THAT IS NOT THERE",
              "The library stage reads nothing the four before it measured. `library_spec` takes an "
              "initiator, a length\nrange and a set of elongator classes — no patch area, "
              "no accession, no residue range.\nThe library is the same library whatever the "
              "surface stage found, so \"a focused library\nover that surface\" states an "
              "intent and not a data path.",
              BREAK, PALEB, 0.0, 0.47 * 2 + 0.06, title_size=10)
    flow.edge("", colour=BREAK, dashed=True, pad=0.007)
    flow.row("5 · chemistry.library_spec",
             f"local tables   ·   {st_lib['seconds']} s\n"
             "in:  nothing from upstream",
             f"{lib['initiator']} initiator, {lib['random_region'][0]}–"
             f"{lib['random_region'][1]} randomised positions\n"
             f"over {lib['alphabet_size']} building blocks, "
             f"{lib['closure_residue']} closes the ring by thioether\n"
             f"{lib['theoretical_diversity']:,} distinct sequences allowed",
             ARM1, SURFACE, PALE1)
    arm1_bottom = flow.y - 0.008
    _band(ax, arm1_top, arm1_bottom, "ARM 1",
          "the onset\nco-scientist\n\nfive stages,\n"
          f"{run['seconds']:.0f} s end\nto end, live\nagainst public\nendpoints",
          ARM1, PALE1)
    flow.y = arm1_bottom
    flow.edge(f"how many positions the library randomises  "
              f"({lib['random_region'][0]}–{lib['random_region'][1]})\n"
              "— the only number that crosses from one arm to the other", colour=ARM2)

    arm2_top = flow.y + 0.006
    flow.row("6 · the go / no-go, before any assay",
             "in:  how many positions the library varies, and whether the\n"
             "      candidates vary at the same positions as the members\n"
             "      already measured.   No assay. No structure. No target.",
             f"candidates share the measured positions  →   {gain_shared}\n"
             f"candidates share none of them               →   {gain_unshared}\n"
             "gain over random from the same batch, on 128 measurements",
             ARM2, SURFACE, PALE2)
    flow.edge("a decision, before anything is synthesised:\n"
              "build this library, or redesign it so the positions overlap", colour=ARM2)
    flow.node("WHAT WOULD CLOSE THE LOOP, AND IS NOT WIRED EITHER",
              "Measured members come back from a campaign as the first round a model is "
              "fitted on,\nand the go/no-go runs on the design that produced them. Eight "
              "registry fields are that\ninterface, named exactly, and all eight are empty "
              "for all 29 blocks: 232 cells of 232.",
              BREAK, PALEB, 0.0, 0.47 * 2 + 0.06, title_size=10)
    _band(ax, arm2_top, flow.y - 0.004, "ARM 2",
          "will a model\nfitted on the\nfirst round be\nworth acting on?\n\n"
          "answerable at\ndesign time, and\nthe strongest\nresult here", ARM2, PALE2)
    return fig
