"""Ad-hoc check: run hierarchical dedup against a synthesized imatinib candidate list.

Not a pytest test — just a script Claude runs to inspect LLM behavior on a
non-rosiglitazone case. Delete after inspection.
"""

import asyncio
import logging

from indication_scout.agents.supervisor.candidate_dedup import run_hierarchical_dedup

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

CANDIDATES = [
    # Oncology — has hierarchical chain: CML ⊂ chronic leukemia ⊂ leukemia
    ("chronic myeloid leukemia", "competitor", "EFO_0000339"),
    ("chronic leukemia", "mechanism", "EFO_0000095"),
    ("leukemia", "mechanism", "EFO_0000565"),
    # Sibling under leukemia — MUST NOT be dropped (imatinib has no AML activity
    # but it's a sibling of CML, not a sub/super-type).
    ("acute myeloid leukemia", "mechanism", "EFO_0000222"),
    # Hierarchical case in a different organ system: GIST ⊂ soft tissue sarcoma
    ("gastrointestinal stromal tumor", "competitor", "EFO_0000559"),
    ("soft tissue sarcoma", "mechanism", "EFO_0006791"),
    # Sibling — different sarcoma subtype, MUST NOT be dropped just because it's
    # also a sarcoma.
    ("synovial sarcoma", "mechanism", "EFO_0002690"),
    # Synonym pair the LLM should collapse
    ("hypereosinophilic syndrome", "competitor", "Orphanet_3260"),
    ("idiopathic hypereosinophilic syndrome", "mechanism", "Orphanet_168956"),
    # Unrelated control — should be left alone
    ("systemic sclerosis", "mechanism", "EFO_0000717"),
]

DRUG = "imatinib"
TARGETS = [("BCR", "INHIBITOR"), ("ABL1", "INHIBITOR"), ("KIT", "INHIBITOR"), ("PDGFRA", "INHIBITOR")]


async def main() -> None:
    out = await run_hierarchical_dedup(
        drug_name=DRUG,
        mechanism_targets=TARGETS,
        candidates=CANDIDATES,
    )
    print(f"\n=== {len(out.decisions)} DECISIONS ===")
    for d in out.decisions:
        print(f"\nsurvivor: {d.survivor!r}")
        print(f"dropped:  {d.dropped}")
        print(f"reason:   {d.reason}")

    all_dropped = {n for d in out.decisions for n in d.dropped}
    print("\n=== CHECKS ===")
    print(f"AML in dropped (BAD if True): {'acute myeloid leukemia' in all_dropped}")
    print(f"Synovial sarcoma in dropped (BAD if True): {'synovial sarcoma' in all_dropped}")
    print(f"Systemic sclerosis in dropped (BAD if True): {'systemic sclerosis' in all_dropped}")
    leuk_overlap = all_dropped & {"chronic myeloid leukemia", "chronic leukemia", "leukemia"}
    print(f"Leukemia chain dropped (expect 1-2 of 3): {leuk_overlap}")
    sarc_overlap = all_dropped & {"gastrointestinal stromal tumor", "soft tissue sarcoma"}
    print(f"GIST/STS dropped (expect 1 of 2): {sarc_overlap}")
    hes_overlap = all_dropped & {"hypereosinophilic syndrome", "idiopathic hypereosinophilic syndrome"}
    print(f"HES synonym dropped (expect 1 of 2): {hes_overlap}")


asyncio.run(main())
