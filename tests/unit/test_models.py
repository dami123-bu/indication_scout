"""Unit tests for domain models."""

from indication_scout.models.drug import Drug, DrugActivity
from indication_scout.models.indication import DiseaseIndication, Indication
from indication_scout.models.target import Target


class TestTarget:
    """Tests for Target model."""

    def test_target_generates_uuid(self):
        """Target should auto-generate an id if not provided."""
        target = Target(symbol="EGFR")
        assert target.id is not None
        assert len(target.id) == 36  # UUID format

    def test_target_all_fields_optional_except_implicit_id(self):
        """Target can be created with no fields (id auto-generated)."""
        target = Target()
        assert target.id is not None
        assert target.ensembl_id is None
        assert target.symbol is None
        assert target.name is None


class TestIndication:
    """Tests for Indication and DiseaseIndication models."""

    def test_indication_requires_name(self):
        """Indication requires a name field."""
        indication = Indication(name="headache")
        assert indication.name == "headache"
        assert indication.id is not None

    def test_disease_indication_inherits_from_indication(self):
        """DiseaseIndication should be an Indication subclass."""
        disease = DiseaseIndication(name="migraine", efo_id="EFO_0003821")
        assert isinstance(disease, Indication)
        assert disease.name == "migraine"
        assert disease.efo_id == "EFO_0003821"

    def test_disease_indication_therapeutic_areas_default_empty(self):
        """therapeutic_areas should default to empty list."""
        disease = DiseaseIndication(name="migraine")
        assert disease.therapeutic_areas == []


class TestDrug:
    """Tests for Drug model."""

    def test_drug_requires_generic_name(self):
        """Drug requires generic_name."""
        drug = Drug(generic_name="aspirin")
        assert drug.generic_name == "aspirin"
        assert drug.id is not None

    def test_drug_activities_default_empty(self):
        """activities should default to empty list."""
        drug = Drug(generic_name="aspirin")
        assert drug.activities == []

    def test_drug_boolean_defaults(self):
        """is_approved and has_been_withdrawn default to False."""
        drug = Drug(generic_name="aspirin")
        assert drug.is_approved is False
        assert drug.has_been_withdrawn is False


class TestDrugActivity:
    """Tests for DrugActivity model."""

    def test_drug_activity_all_fields_optional(self):
        """DrugActivity can be created with no fields."""
        activity = DrugActivity()
        assert activity.description is None
        assert activity.action_type is None
        assert activity.target is None
        assert activity.indication is None

    def test_drug_activity_with_target(self):
        """DrugActivity can hold a target."""
        target = Target(ensembl_id="ENSG00000146648", symbol="EGFR")
        activity = DrugActivity(
            description="Inhibitor of COX enzymes",
            target=target,
        )
        assert activity.target.symbol == "EGFR"

    def test_drug_activity_with_indication(self):
        """DrugActivity can hold an indication."""
        indication = DiseaseIndication(name="migraine", efo_id="EFO_0003821")
        activity = DrugActivity(indication=indication)
        assert activity.indication.name == "migraine"
