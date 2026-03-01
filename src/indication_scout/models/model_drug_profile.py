"""Pydantic model for a drug's profile used by the RAG pipeline."""

from pydantic import BaseModel

from indication_scout.models.model_chembl import ATCDescription
from indication_scout.models.model_open_targets import RichDrugData


class DrugProfile(BaseModel):
    """Structured drug profile consumed by expand_search_terms.

    Built from RichDrugData + pre-fetched ATCDescription objects via
    the from_rich_drug_data factory classmethod.
    """

    name: str
    synonyms: list[str]
    target_gene_symbols: list[str]
    mechanisms_of_action: list[str]
    atc_codes: list[str]
    atc_descriptions: list[str]
    drug_type: str

    @classmethod
    def from_rich_drug_data(
        cls,
        rich: RichDrugData,
        atc_descriptions: list[ATCDescription],
    ) -> "DrugProfile":
        """Build a DrugProfile from a RichDrugData and pre-fetched ATC descriptions.

        Args:
            rich: Combined drug + target data from Open Targets.
            atc_descriptions: ATCDescription objects for each ATC code on the drug,
                fetched by the caller via ChEMBLClient.get_atc_description.
        """
        synonyms = list(dict.fromkeys(rich.drug.synonyms + rich.drug.trade_names))

        target_gene_symbols = list(
            dict.fromkeys(t.symbol for t in rich.targets)
        )

        mechanisms_of_action = list(
            dict.fromkeys(
                dt.mechanism_of_action for dt in rich.drug.targets
            )
        )

        atc_desc_strings = list(
            dict.fromkeys(
                desc
                for atc in atc_descriptions
                for desc in (atc.level3_description, atc.level4_description)
            )
        )

        return cls(
            name=rich.drug.name,
            synonyms=synonyms,
            target_gene_symbols=target_gene_symbols,
            mechanisms_of_action=mechanisms_of_action,
            atc_codes=rich.drug.atc_classifications,
            atc_descriptions=atc_desc_strings,
            drug_type=rich.drug.drug_type,
        )
