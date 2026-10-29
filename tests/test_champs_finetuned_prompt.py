from coda.inference.champs_finetuned.prompt import \
    render_profile_abstraction, render_profile_evidence
from coda.metadata import Profile


def test_profile_evidence():
    profile = Profile.from_dict({
        "age": {"value": 11, "unit": "months"}, "sex": "female",
        "date_of_death": "2026-10-14", "birth_weight_g": 2400,
        "weight_kg_death": 6.1, "muac_cm_death": 11.2})
    assert render_profile_evidence(profile).splitlines() == [
        "- Sex: female",
        "- De-Identified Date of Death: 2026-10-14",
        "- Age in Years: 0",
        "- Age in Months: 11",
        "- Age Grouping of the CHAMPS Case: "
        "infant (28 days to less than 12 months)",
        "- Weight in kilograms: 6.1",
        "- Mid-upper arm circumference in centimeters: 11.2",
        "- Birth weight (grams): 2400.0",
    ]


def test_stillbirth_and_hour_evidence():
    stillbirth = Profile.from_dict({"stillbirth": True,
                                    "age": {"value": 3, "unit": "days"}})
    assert render_profile_evidence(stillbirth) == \
        "- Age Grouping of the CHAMPS Case: stillbirth"
    hours = Profile.from_dict({"age": {"value": 30, "unit": "hours"}})
    assert "- Age in days: 1.0\n- Age in hours: 6.0" in \
        render_profile_evidence(hours)
    months = Profile.from_dict({"age": {"value": 30, "unit": "months"}})
    assert "- Age in Years: 2\n- Age in Months: 6" in \
        render_profile_evidence(months)
    assert render_profile_evidence(None) == ""


def test_profile_abstraction():
    profile = Profile.from_dict({"hiv_status": "exposed",
                                 "weight_kg_admission": 6.0,
                                 "muac_cm_admission": 11.5})
    assert render_profile_abstraction(profile).splitlines() == [
        "HIV exposure: Yes",
        "Weight (kilograms): 6",
        "Mid-upper arm circumference (cm): 11.5",
    ]
    unknown = Profile.from_dict({"hiv_status": "unknown", "sex": "male"})
    assert render_profile_abstraction(unknown) == ""
