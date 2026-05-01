"""Format a SupervisorOutput as a Markdown report."""

from datetime import datetime

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.agents.clinical_trials.clinical_trials_output import ClinicalTrialsOutput
from indication_scout.agents.clinical_trials.clinical_trials_tools import _classify_stop_reason
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


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


def _fmt_clinical_trials(ct: ClinicalTrialsOutput) -> str:
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
                category = f" [{classified}]" if classified not in {"unknown", "other"} else ""
                lines.append(f"- [{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id}){title} ({phase}){category}{reason}")

    if ct.landscape and ct.landscape.competitors:
        lines.append(f"\n**Competitive landscape ({len(ct.landscape.competitors)} competitors):**")
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
            lines.append(f"- **{c.target_symbol} ({c.action_type}) → {c.disease_name}**")
            if c.disease_description:
                lines.append(f"  - {c.disease_description}")
            if c.target_function:
                lines.append(f"  - _Target function:_ {c.target_function}")

    return "\n".join(lines)


def format_report(output: SupervisorOutput) -> str:
    """Render a SupervisorOutput as a Markdown string."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    drug = output.drug_name or "Unknown drug"

    lines: list[str] = [
        f"# IndicationScout Report: {drug}",
        f"_Generated {now}_",
        "",
        "---",
        "",
    ]

    # Summary
    lines += [
        "## Summary",
        "",
        output.summary if output.summary else "_No summary produced._",
        "",
        "---",
        "",
    ]

    # Candidate diseases
    lines += ["## Candidate Diseases", ""]
    if output.candidates:
        for c in output.candidates:
            lines.append(f"- {c}")
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
            lines += [f"### {finding.disease} _(source: {finding.source})_", ""]

            if finding.literature:
                lines += ["#### Literature", "", _fmt_literature(finding.literature), ""]

            if finding.clinical_trials:
                lines += ["#### Clinical Trials", "", _fmt_clinical_trials(finding.clinical_trials), ""]

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
                mech_only_uninvestigated.append(c.disease_name)

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
