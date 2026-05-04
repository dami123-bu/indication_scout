"""Format a SupervisorOutput as a Markdown report."""

import re
from datetime import datetime

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import ClinicalTrialsOutput
from indication_scout.agents.clinical_trials.clinical_trials_tools import _classify_stop_reason
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


def _title_case_disease(name: str) -> str:
    """Capitalize the first letter of each whitespace-separated word, leaving the
    rest of each word untouched so acronyms (e.g. NSCLC) and possessives (e.g.
    Alzheimer's) are preserved."""
    return " ".join(w[:1].upper() + w[1:] if w else w for w in name.split(" "))


def _fmt_literature(lit: LiteratureOutput) -> str:
    lines: list[str] = []

    if lit.evidence_summary:
        es = lit.evidence_summary
        lines.append(f"**Evidence strength:** {es.strength}")
        lines.append(f"**Study count:** {es.study_count}")
        if es.summary:
            lines.append(f"\n{es.summary}")
        if es.key_findings:
            lines.append("\n**Key findings:**")
            for finding in es.key_findings:
                lines.append(f"- {finding}")
        if es.supporting_pmids:
            pmid_links = ", ".join(
                f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                for pmid in es.supporting_pmids
            )
            lines.append(f"\n**Supporting PMIDs:** {pmid_links}")
    else:
        lines.append("_No evidence summary available._")

    return "\n".join(lines)


def _fmt_clinical_trials(ct: ClinicalTrialsOutput, indication: str = "") -> str:
    lines: list[str] = []

    if ct.summary:
        lines.append(ct.summary)

    if ct.approval:
        ap = ct.approval
        if ap.is_approved:
            matched = f" ({ap.matched_indication})" if ap.matched_indication else ""
            lines.append(f"\n**FDA approval:** Approved{matched}")
        elif ap.label_found:
            lines.append("\n**FDA approval:** Not found on FDA label for this indication")
        else:
            names = ", ".join(ap.drug_names_checked) if ap.drug_names_checked else "drug"
            lines.append(f"\n**FDA approval:** No FDA label found for {names} — status undetermined")

    if ct.search:
        s = ct.search
        lines.append(f"\n**Trial activity:** {s.total_count} total trial(s) for this pair")
        if s.by_status:
            status_bits = [f"{count} {status.lower().replace('_', ' ')}" for status, count in s.by_status.items() if count]
            if status_bits:
                lines.append(f"- {', '.join(status_bits)}")
        if s.total_count == 0:
            lines.append("- _Whitespace: no trials found for this drug × indication pair._")

    if ct.completed:
        c = ct.completed
        lines.append(f"\n**Completed trials ({c.total_count} total):**")
        for trial in c.trials[:10]:
            phase = trial.phase or "Unknown phase"
            status = trial.overall_status or ""
            lines.append(f"- [{trial.nct_id}](https://clinicaltrials.gov/study/{trial.nct_id}) — {trial.title} ({phase}{', ' + status if status else ''})")

    if ct.terminated:
        term = ct.terminated
        if term.total_count:
            lines.append(f"\n**Terminated trials ({term.total_count}):**")
            for t in term.trials[:10]:
                reason = f" — *{t.why_stopped}*" if t.why_stopped else ""
                title = f" {t.title}" if t.title else ""
                phase = t.phase or "Unknown phase"
                classified = _classify_stop_reason(t.why_stopped)
                category = (
                    f" [{classified}]"
                    if classified not in {"unknown", "other"}
                    and classified != t.why_stopped
                    else ""
                )
                lines.append(f"- [{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id}){title} ({phase}){category}{reason}")

    if ct.landscape and ct.landscape.competitors:
        scope = f" for {_title_case_disease(indication)}" if indication else ""
        lines.append(
            f"\n**Competitive landscape{scope} "
            f"({len(ct.landscape.competitors)} competitors):**"
        )
        for comp in ct.landscape.competitors[:10]:
            lines.append(f"- {comp.drug_name} ({comp.sponsor}) — {comp.max_phase}, {comp.trial_count} trial(s)")

    if not lines:
        lines.append("_No clinical trials data available._")

    return "\n".join(lines)


def _fmt_mechanism(mech: MechanismOutput) -> str:
    lines: list[str] = []

    if mech.summary:
        lines.append(mech.summary)

    if mech.drug_targets:
        targets = ", ".join(sorted(mech.drug_targets.keys()))
        lines.append(f"\n**Molecular targets:** {targets}")

    if mech.mechanisms_of_action:
        lines.append("\n**Mechanisms of action:**")
        for moa in mech.mechanisms_of_action:
            syms = ", ".join(moa.target_symbols) if moa.target_symbols else "—"
            lines.append(f"- {moa.mechanism_of_action} ({moa.action_type}) → {syms}")

    if mech.candidates:
        lines.append("\n**Repurposing candidates:**")
        for c in mech.candidates:
            lines.append(f"- **{c.target_symbol} ({c.action_type}) → {_title_case_disease(c.disease_name)}**")
            if c.disease_description:
                lines.append(f"  - {c.disease_description}")
            if c.target_function:
                lines.append(f"  - _Target function:_ {c.target_function}")

    return "\n".join(lines)


def _title_case_known_diseases(text: str, disease_names: list[str]) -> str:
    """Replace every case-insensitive occurrence of each known disease name in ``text``
    with its title-cased form. Longest names first so multi-word names aren't shadowed
    by their substrings (e.g. "non-small cell lung cancer" before "lung cancer")."""
    if not text or not disease_names:
        return text
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in disease_names:
        if not name:
            continue
        key = name.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    unique_names.sort(key=len, reverse=True)
    for name in unique_names:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub(_title_case_disease(name), text)
    return text


def _splice_blurbs_into_summary(
    summary: str, findings: list[CandidateFindings]
) -> str:
    """Replace each ranked summary line's structured tail with the matching blurb.

    The supervisor's summary string is a ranked list of the form
    `N. <disease> — literature: ..., trials: ...; FDA approval: ...`. For each line
    that matches a finding with a non-empty blurb, the structured tail (everything
    from the em-dash onward) is stripped and replaced with the blurb on the next
    line. Lines that don't match any finding (e.g. the trailing "Closed signals:"
    line) and lines without an em-dash are passed through unchanged. Disease
    matching is case-insensitive on the disease name only.
    """
    blurb_by_disease = {
        f.disease.lower().strip(): f.blurb.strip()
        for f in findings
        if f.blurb and f.blurb.strip()
    }
    if not blurb_by_disease:
        return summary

    # Group 1: rank prefix ("N. "). Group "head": disease portion (before em-dash).
    # The em-dash separator and structured tail are dropped on a successful match.
    rank_line = re.compile(r"^(\s*\d+\.\s+)(?P<head>.+?)\s+—\s+.+$")
    out_lines: list[str] = []
    for line in summary.splitlines():
        m = rank_line.match(line)
        if m is None:
            out_lines.append(line)
            continue
        head = m.group("head")
        head_lower = head.lower()
        match_key: str | None = None
        # Longest-match avoids "lung cancer" stealing a match meant for
        # "non-small cell lung cancer".
        for key in sorted(blurb_by_disease.keys(), key=len, reverse=True):
            if key and key in head_lower:
                match_key = key
                break
        if match_key is None:
            out_lines.append(line)
            continue
        blurb = blurb_by_disease.pop(match_key)
        prefix = m.group(1)
        out_lines.append(f"{prefix}{_title_case_disease(head)}")
        out_lines.append(f"   _{blurb}_")
    return "\n".join(out_lines)


def format_report(output: SupervisorOutput) -> str:
    """Render a SupervisorOutput as a Markdown string."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    drug = output.drug_name or "Unknown drug"

    lines: list[str] = [
        f"# IndicationScout Report: {_title_case_disease(drug)}",
        f"_Generated {now}_",
        "",
        "---",
        "",
    ]

    # Summary — splice each top-5 candidate's blurb directly under its ranked line,
    # then title-case any known disease names that appear inside the LLM-generated text.
    if output.summary:
        summary_text = _splice_blurbs_into_summary(output.summary, output.findings)
        known_diseases = [f.disease for f in output.findings] + list(output.candidates or [])
        summary_text = _title_case_known_diseases(summary_text, known_diseases)
    else:
        summary_text = "_No summary produced._"
    lines += [
        "## Summary",
        "",
        summary_text,
        "",
        "_Note: trial counts in this summary reflect ClinicalTrials.gov only and may "
        "undercount activity registered in ex-US registries (e.g. jRCT, ChiCTR, "
        "EU-CTR, ANZCTR). Studies cited in the literature section may reference "
        "trials in those registries that are not represented in the trial counts above._",
        "",
        "---",
        "",
    ]

    # Candidate diseases
    lines += ["## Candidate Diseases", ""]
    if output.candidates:
        lines.append(
            "_Note: not every candidate listed here is investigated in depth. "
            "Only diseases with a section under **Candidate Findings** below have "
            "literature and clinical-trial evidence pulled for this run._"
        )
        lines.append("")
        for c in output.candidates:
            lines.append(f"- {_title_case_disease(c)}")
    else:
        lines.append("_No candidates surfaced._")
    lines += ["", "---", ""]

    # Mechanism
    lines += ["## Mechanistic Analysis", ""]
    if output.mechanism:
        lines.append(_fmt_mechanism(output.mechanism))
    else:
        lines.append("_Mechanism analysis not run._")
    lines += ["", "---", ""]

    # Per-candidate findings
    lines += ["## Candidate Findings", ""]
    if output.findings:
        # Build set of investigated disease keys so we can list any
        # mechanism candidates that were promoted but not investigated below.
        investigated_keys = {f.disease.lower().strip() for f in output.findings}

        for finding in output.findings:
            lines += [f"## {_title_case_disease(finding.disease)} _(source: {finding.source})_", ""]

            if finding.literature:
                lines += [f"### Literature — {_title_case_disease(finding.disease)}", "", _fmt_literature(finding.literature), ""]

            if finding.clinical_trials:
                lines += ["### Clinical Trials", "", _fmt_clinical_trials(finding.clinical_trials, finding.disease), ""]

            lines.append("---")
            lines.append("")

        # Surface mechanism candidates that were promoted to the allowlist
        # but not selected for deep investigation in the findings above.
        mech_only_uninvestigated: list[str] = []
        if output.mechanism and output.mechanism.candidates:
            seen: set[str] = set()
            for c in output.mechanism.candidates:
                key = c.disease_name.lower().strip()
                if not key or key in seen or key in investigated_keys:
                    continue
                seen.add(key)
                mech_only_uninvestigated.append(_title_case_disease(c.disease_name))

        if mech_only_uninvestigated:
            lines += [
                "### Other mechanism candidates (promoted, not investigated)",
                "",
                "These diseases were surfaced by the mechanism agent and added to the "
                "investigation allowlist, but the supervisor did not select them for deep "
                "literature / clinical-trials analysis in this run.",
                "",
            ]
            for name in mech_only_uninvestigated:
                lines.append(f"- {name}")
            lines += ["", "---", ""]
    else:
        lines.append("_No candidate findings produced._")

    return "\n".join(lines)
