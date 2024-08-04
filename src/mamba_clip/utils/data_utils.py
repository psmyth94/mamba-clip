# type: ignore
import random

import numpy as np
import pandas as pd
import pandas.api.types


def get_patient_information(data, is_eval=False):
    templates = {
        "all_present": [
            "The patient, a {age} year-old {sex}, presented with a lesion located on the {anatom_site_general}.",
            "A {age} year-old {sex} patient presented with a lesion on the {anatom_site_general}.",
            "{age} years old and {sex}, the patient had a lesion located on the {anatom_site_general}.",
            "The patient, aged {age} years and {sex}, was found to have a lesion on the {anatom_site_general}.",
            "{age} years old, the patient, a {sex}, had a lesion on the {anatom_site_general}.",
            "An {age} year-old {sex} presented with a lesion on the {anatom_site_general}.",
            "A lesion was found on the {anatom_site_general} of the {age} year-old {sex} patient.",
            "The patient, {age} years old and {sex}, had a lesion on the {anatom_site_general}.",
            "The {sex} patient, aged {age} years, presented with a lesion on the {anatom_site_general}.",
            "At {age} years old, the {sex} patient had a lesion located on the {anatom_site_general}.",
            "A lesion was identified on the {anatom_site_general} of the {age} year-old {sex}.",
            "{age} year-old {sex} with a lesion on the {anatom_site_general}.",
        ],
        "age_sex": [
            "The patient, a {age} year-old {sex}, presented with a lesion.",
            "A {age} year-old {sex} patient presented with a lesion.",
            "{age} years old and {sex}, the patient had a lesion.",
            "The patient, aged {age} years and {sex}, had a lesion.",
            "An {age} year-old {sex} was found with a lesion.",
            "A lesion was identified in the {age} year-old {sex} patient.",
        ],
        "age_anatom": [
            "The patient, approximately {age} years old, presented with a lesion located on the {anatom_site_general}.",
            "Approximately {age} years old, the patient had a lesion on the {anatom_site_general}.",
            "The patient had a lesion located on the {anatom_site_general}, and was approximately {age} years old.",
            "A lesion was found on the {anatom_site_general} of a patient approximately {age} years old.",
            "At approximately {age} years old, the patient had a lesion on the {anatom_site_general}.",
            "The lesion on the {anatom_site_general} was found in a patient around {age} years old.",
        ],
        "sex_anatom": [
            "The patient, {sex}, presented with a lesion located on the {anatom_site_general}.",
            "A {sex} patient presented with a lesion on the {anatom_site_general}.",
            "{sex}, the patient had a lesion on the {anatom_site_general}.",
            "A lesion was located on the {anatom_site_general} of the {sex} patient.",
            "The {sex} patient had a lesion on the {anatom_site_general}.",
            "On the {anatom_site_general} of the {sex} patient, a lesion was found.",
        ],
        "age_only": [
            "The patient, approximately {age} years old, presented with a lesion.",
            "Approximately {age} years old, the patient had a lesion.",
            "The patient had a lesion and was approximately {age} years old.",
            "At around {age} years old, the patient was found to have a lesion.",
            "A lesion was identified in a patient approximately {age} years old.",
            "The lesion was found in a patient aged around {age} years.",
        ],
        "sex_only": [
            "The patient, {sex}, presented with a lesion.",
            "A {sex} patient presented with a lesion.",
            "{sex}, the patient had a lesion.",
            "A lesion was found in the {sex} patient.",
            "The {sex} patient was diagnosed with a lesion.",
            "A lesion was identified in the {sex} patient.",
        ],
        "anatom_only": [
            "The patient presented with a lesion located on the {anatom_site_general}.",
            "The patient had a lesion on the {anatom_site_general}.",
            "A lesion was located on the patient's {anatom_site_general}.",
            "The lesion was found on the {anatom_site_general}.",
            "On the {anatom_site_general}, the patient had a lesion.",
            "A lesion was identified on the patient's {anatom_site_general}.",
        ],
    }

    age = int(data["age_approx"]) if not pd.isna(data["age_approx"]) else None
    sex = data["sex"] if not pd.isna(data["sex"]) else None
    anatom_site_general = (
        data["anatom_site_general"]
        if not pd.isna(data["anatom_site_general"])
        else None
    )
    if is_eval:
        templates = {k: [v[0]] for k, v in templates.items()}
    if age and sex and anatom_site_general:
        template = np.random.choice(templates["all_present"])
    elif age and sex:
        template = np.random.choice(templates["age_sex"])
    elif age and anatom_site_general:
        template = np.random.choice(templates["age_anatom"])
    elif sex and anatom_site_general:
        template = np.random.choice(templates["sex_anatom"])
    elif age:
        template = np.random.choice(templates["age_only"])
    elif sex:
        template = np.random.choice(templates["sex_only"])
    elif anatom_site_general:
        template = np.random.choice(templates["anatom_only"])
    else:
        return None

    return template.format(age=age, sex=sex, anatom_site_general=anatom_site_general)


def get_hue_info(data, is_eval=False):
    templates = [
        "The hue inside the lesion was measured at {hue}.",
        "Hue inside the lesion was recorded as {hue}.",
        "The inside hue of the lesion measured {hue}.",
        "Measured hue inside the lesion was {hue}.",
        "The lesion's internal hue was {hue}.",
        "Hue measurement inside the lesion was {hue}.",
        "The internal hue of the lesion recorded {hue}.",
        "The lesion showed an internal hue of {hue}.",
        "Inside the lesion, the hue was measured at {hue}.",
        "The hue value inside the lesion was {hue}.",
    ]
    hue = round(data["tbp_lv_H"], 1)
    return (
        np.random.choice(templates).format(hue=hue)
        if not is_eval
        else templates[0].format(hue=hue)
    )


def get_area_info(data, is_eval=False):
    templates = [
        "The lesion covered an area of {area} mm squared.",
        "Lesion area was measured at {area} mm squared.",
        "An area of {area} mm squared was covered by the lesion.",
        "The lesion spanned {area} mm squared.",
        "Measured area of the lesion was {area} mm squared.",
        "The lesion's area was {area} mm squared.",
        "The area of the lesion was measured to be {area} mm squared.",
        "Covering an area of {area} mm squared, the lesion was observed.",
        "An observed lesion spanned {area} mm squared.",
        "The lesion's coverage area was recorded as {area} mm squared.",
    ]
    area = round(data["tbp_lv_areaMM2"], 1)
    return (
        np.random.choice(templates).format(area=area)
        if not is_eval
        else templates[0].format(area=area)
    )


def get_border_jaggedness_info(data, is_eval=False):
    templates = [
        "The border jaggedness, characterized by the area-perimeter ratio, was {jaggedness}.",
        "Border jaggedness, defined by the area-perimeter ratio, measured {jaggedness}.",
        "The lesion's border jaggedness was {jaggedness}.",
        "Measured border jaggedness was {jaggedness}.",
        "The ratio defining the border's jaggedness was {jaggedness}.",
        "The lesion's jagged border ratio was {jaggedness}.",
        "Jaggedness of the lesion's border was {jaggedness}.",
        "The lesion's border had a jaggedness ratio of {jaggedness}.",
        "The area-perimeter ratio of the lesion's border was {jaggedness}.",
        "The border jaggedness ratio was recorded as {jaggedness}.",
    ]
    jaggedness = round(data["tbp_lv_area_perim_ratio"], 1)
    return (
        np.random.choice(templates).format(jaggedness=jaggedness)
        if not is_eval
        else templates[0].format(jaggedness=jaggedness)
    )


def get_color_irregularity_info(data, is_eval=False):
    templates = [
        "Color irregularity within the lesion was {irregularity}.",
        "The lesion's color irregularity was measured at {irregularity}.",
        "Measured color irregularity inside the lesion was {irregularity}.",
        "Irregularity of color within the lesion was {irregularity}.",
        "The internal color irregularity of the lesion measured {irregularity}.",
        "The lesion showed color irregularity of {irregularity}.",
        "The lesion had an internal color irregularity measured at {irregularity}.",
        "The irregularity in the lesion's color was {irregularity}.",
        "A color irregularity of {irregularity} was observed within the lesion.",
        "Internal color irregularity of the lesion was recorded as {irregularity}.",
    ]
    irregularity = round(data["tbp_lv_color_std_mean"], 1)
    return (
        np.random.choice(templates).format(irregularity=irregularity)
        if not is_eval
        else templates[0].format(irregularity=irregularity)
    )


def get_contrast_info(data, is_eval=False):
    templates = [
        "The overall contrast of the lesion relative to the surrounding skin was {contrast}.",
        "Lesion contrast compared to surrounding skin was {contrast}.",
        "The lesion had a contrast value of {contrast} relative to the surrounding skin.",
        "Measured contrast between lesion and surrounding skin was {contrast}.",
        "The lesion's contrast with the surrounding skin was {contrast}.",
        "Contrast of the lesion relative to surrounding skin was {contrast}.",
        "The contrast measurement of the lesion to the surrounding skin was {contrast}.",
        "The lesion exhibited a contrast of {contrast} against the surrounding skin.",
        "A contrast of {contrast} was noted between the lesion and surrounding skin.",
        "The surrounding skin and lesion contrast was recorded as {contrast}.",
    ]
    contrast = round(data["tbp_lv_deltaLBnorm"], 1)
    return (
        np.random.choice(templates).format(contrast=contrast)
        if not is_eval
        else templates[0].format(contrast=contrast)
    )


def get_eccentricity_info(data, is_eval=False):
    templates = [
        "The eccentricity of the lesion was noted to be {eccentricity}.",
        "Lesion eccentricity measured {eccentricity}.",
        "Eccentricity of the lesion was {eccentricity}.",
        "Measured eccentricity of the lesion was {eccentricity}.",
        "The lesion's eccentricity was recorded as {eccentricity}.",
        "Eccentricity of the lesion was observed to be {eccentricity}.",
        "The lesion showed an eccentricity of {eccentricity}.",
        "An eccentricity value of {eccentricity} was recorded for the lesion.",
        "The lesion had an eccentricity measurement of {eccentricity}.",
        "The recorded eccentricity of the lesion was {eccentricity}.",
    ]
    eccentricity = round(data["tbp_lv_eccentricity"], 2)
    return (
        np.random.choice(templates).format(eccentricity=eccentricity)
        if not is_eval
        else templates[0].format(eccentricity=eccentricity)
    )


def get_location_info(data, is_eval=False):
    templates = [
        "Anatomical location was simplified as {location}.",
        "The lesion's anatomical location was categorized as {location}.",
        "Simplified anatomical location of the lesion was {location}.",
        "The lesion was located in the {location} region.",
        "Location of the lesion was recorded as {location}.",
        "The lesion was found in the {location} anatomical region.",
        "The anatomical categorization of the lesion's location was {location}.",
        "Recorded anatomical location of the lesion was {location}.",
        "The lesion was anatomically located in the {location} region.",
        "Anatomical classification of the lesion's location was {location}.",
    ]
    location = data["tbp_lv_location_simple"]
    return (
        np.random.choice(templates).format(location=location)
        if not is_eval
        else templates[0].format(location=location)
    )


def get_minor_axis_info(data, is_eval=False):
    templates = [
        "The smallest diameter of the lesion was {minor_axis} mm.",
        "Lesion's minor axis measured {minor_axis} mm.",
        "The lesion had a minor axis of {minor_axis} mm.",
        "Measured smallest diameter of the lesion was {minor_axis} mm.",
        "The minor axis of the lesion was {minor_axis} mm.",
        "The lesion's smallest diameter was {minor_axis} mm.",
        "Minor axis of the lesion was recorded as {minor_axis} mm.",
        "The smallest measured diameter of the lesion was {minor_axis} mm.",
        "The lesion showed a minor axis measurement of {minor_axis} mm.",
        "A minor axis of {minor_axis} mm was recorded for the lesion.",
    ]
    minor_axis = round(data["tbp_lv_minorAxisMM"], 1)
    return (
        np.random.choice(templates).format(minor_axis=minor_axis)
        if not is_eval
        else templates[0].format(minor_axis=minor_axis)
    )


def get_nevi_confidence_info(data, is_eval=False):
    templates = [
        "The confidence score that the lesion is a nevus was {confidence} out of 100.",
        "Nevus confidence score of the lesion was {confidence} out of 100.",
        "The lesion had a nevus confidence score of {confidence} out of 100.",
        "Measured nevus confidence score was {confidence} out of 100.",
        "Confidence that the lesion is a nevus was {confidence} out of 100.",
        "The lesion showed a nevus confidence score of {confidence} out of 100.",
        "A nevus confidence score of {confidence} out of 100 was recorded for the lesion.",
        "The lesion's nevus confidence was measured at {confidence} out of 100.",
        "A confidence score of {confidence} out of 100 indicated the lesion is a nevus.",
        "The recorded nevus confidence score of the lesion was {confidence} out of 100.",
    ]
    confidence = int(data["tbp_lv_nevi_confidence"])
    return (
        np.random.choice(templates).format(confidence=confidence)
        if not is_eval
        else templates[0].format(confidence=confidence)
    )


def get_border_irregularity_info(data, is_eval=False):
    templates = [
        "Border irregularity scored {irregularity}.",
        "The lesion's border irregularity was {irregularity}.",
        "Measured border irregularity was {irregularity}.",
        "Irregularity of the lesion's border was {irregularity}.",
        "Border irregularity of the lesion measured {irregularity}.",
        "The recorded border irregularity was {irregularity}.",
        "The lesion showed a border irregularity score of {irregularity}.",
        "A border irregularity score of {irregularity} was recorded for the lesion.",
        "The lesion's border showed an irregularity of {irregularity}.",
        "The measured border irregularity of the lesion was {irregularity}.",
    ]
    irregularity = round(data["tbp_lv_norm_border"], 1)
    return (
        np.random.choice(templates).format(irregularity=irregularity)
        if not is_eval
        else templates[0].format(irregularity=irregularity)
    )


def get_color_variation_info(data, is_eval=False):
    templates = [
        "Color variation scored {variation}.",
        "The lesion's color variation was {variation}.",
        "Measured color variation was {variation}.",
        "Variation in the lesion's color was {variation}.",
        "Color variation of the lesion measured {variation}.",
        "The lesion showed a color variation score of {variation}.",
        "The recorded color variation of the lesion was {variation}.",
        "The lesion had a color variation of {variation}.",
        "A color variation score of {variation} was recorded for the lesion.",
        "The lesion's color variation was measured at {variation}.",
    ]
    variation = round(data["tbp_lv_norm_color"], 1)
    return (
        np.random.choice(templates).format(variation=variation)
        if not is_eval
        else templates[0].format(variation=variation)
    )


def get_perimeter_info(data, is_eval=False):
    templates = [
        "The perimeter of the lesion was {perimeter} mm.",
        "Lesion perimeter measured {perimeter} mm.",
        "The lesion had a perimeter of {perimeter} mm.",
        "Measured perimeter of the lesion was {perimeter} mm.",
        "The lesion's perimeter was recorded as {perimeter} mm.",
        "A perimeter of {perimeter} mm was recorded for the lesion.",
        "The lesion showed a perimeter measurement of {perimeter} mm.",
        "The measured perimeter of the lesion was {perimeter} mm.",
        "The lesion's perimeter measurement was {perimeter} mm.",
        "The lesion was found to have a perimeter of {perimeter} mm.",
    ]
    perimeter = round(data["tbp_lv_perimeterMM"], 1)
    return (
        np.random.choice(templates).format(perimeter=perimeter)
        if not is_eval
        else templates[0].format(perimeter=perimeter)
    )


def get_color_asymmetry_info(data, is_eval=False):
    templates = [
        "Color asymmetry within the lesion was measured at {asymmetry}.",
        "The lesion's color asymmetry was {asymmetry}.",
        "Measured color asymmetry within the lesion was {asymmetry}.",
        "Asymmetry of color within the lesion was {asymmetry}.",
        "The internal color asymmetry of the lesion measured {asymmetry}.",
        "The lesion showed color asymmetry of {asymmetry}.",
        "The lesion had an internal color asymmetry measured at {asymmetry}.",
        "The asymmetry in the lesion's color was {asymmetry}.",
        "A color asymmetry of {asymmetry} was observed within the lesion.",
        "Internal color asymmetry of the lesion was recorded as {asymmetry}.",
    ]
    asymmetry = round(data["tbp_lv_radial_color_std_max"], 1)
    return (
        np.random.choice(templates).format(asymmetry=asymmetry)
        if not is_eval
        else templates[0].format(asymmetry=asymmetry)
    )


def get_assymetry_info(data, is_eval=False):
    templates = {
        "both_present": [
            "Border asymmetry was scored at {tbp_lv_symm_2axis} with an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "The lesion showed a border asymmetry score of {tbp_lv_symm_2axis} and an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "With a border asymmetry score of {tbp_lv_symm_2axis}, the lesion had an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "The lesion had a border asymmetry score of {tbp_lv_symm_2axis} and an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "A border asymmetry score of {tbp_lv_symm_2axis} and an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees were observed in the lesion.",
            "The lesion exhibited a border asymmetry score of {tbp_lv_symm_2axis} and an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "The angle of asymmetry was {tbp_lv_symm_2axis_angle} degrees with a border asymmetry score of {tbp_lv_symm_2axis}.",
            "An asymmetry angle of {tbp_lv_symm_2axis_angle} degrees was noted with a border asymmetry score of {tbp_lv_symm_2axis}.",
            "The lesion showed an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees with a border asymmetry score of {tbp_lv_symm_2axis}.",
        ],
        "symm_only": [
            "Border asymmetry was scored at {tbp_lv_symm_2axis}.",
            "The lesion had a border asymmetry score of {tbp_lv_symm_2axis}.",
            "A border asymmetry score of {tbp_lv_symm_2axis} was observed in the lesion.",
            "The lesion exhibited a border asymmetry score of {tbp_lv_symm_2axis}.",
            "With a border asymmetry score of {tbp_lv_symm_2axis}, the lesion was observed.",
            "The lesion showed a border asymmetry score of {tbp_lv_symm_2axis}.",
            "The lesion had an asymmetry score of {tbp_lv_symm_2axis}.",
            "The lesion's border asymmetry score was {tbp_lv_symm_2axis}.",
            "The border asymmetry score was recorded as {tbp_lv_symm_2axis}.",
        ],
        "angle_only": [
            "The asymmetry angle was {tbp_lv_symm_2axis_angle} degrees.",
            "An asymmetry angle of {tbp_lv_symm_2axis_angle} degrees was noted.",
            "The lesion exhibited an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "The lesion had an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "An asymmetry angle of {tbp_lv_symm_2axis_angle} degrees was observed in the lesion.",
            "The lesion showed an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "The lesion had an asymmetry angle of {tbp_lv_symm_2axis_angle} degrees.",
            "The lesion's asymmetry angle was {tbp_lv_symm_2axis_angle} degrees.",
        ],
    }

    tbp_lv_symm_2axis = (
        data["tbp_lv_symm_2axis"] if not pd.isna(data["tbp_lv_symm_2axis"]) else None
    )
    tbp_lv_symm_2axis_angle = (
        data["tbp_lv_symm_2axis_angle"]
        if not pd.isna(data["tbp_lv_symm_2axis_angle"])
        else None
    )

    if is_eval:
        templates = {k: [v[0]] for k, v in templates.items()}
    if tbp_lv_symm_2axis and tbp_lv_symm_2axis_angle:
        template = np.random.choice(templates["both_present"])
    elif tbp_lv_symm_2axis:
        template = np.random.choice(templates["symm_only"])
    elif tbp_lv_symm_2axis_angle:
        template = np.random.choice(templates["angle_only"])
    else:
        return None

    return template.format(
        tbp_lv_symm_2axis=round(tbp_lv_symm_2axis, 1),
        tbp_lv_symm_2axis_angle=round(tbp_lv_symm_2axis_angle, 1),
    )


def get_target_info(data):
    templates = [
        "The lesion was determined to be {diagnosis}.",
        "Upon investigation, the lesion was classified as {diagnosis}.",
        "The lesion diagnosis was {diagnosis}.",
        "The analysis revealed the lesion to be {diagnosis}.",
        "The lesion was identified as {diagnosis}.",
        "The final diagnosis of the lesion was {diagnosis}.",
        "The lesion was conclusively found to be {diagnosis}.",
        "The lesion's classification was {diagnosis}.",
        "It was concluded that the lesion is {diagnosis}.",
        "The medical examination showed the lesion to be {diagnosis}.",
        "Based on the findings, the lesion was {diagnosis}.",
        "The lesion examination resulted in a diagnosis of {diagnosis}.",
        "The lesion, as shown in the image, was diagnosed as {diagnosis}.",
        "According to the description and image, the lesion is {diagnosis}.",
        "In the provided picture, the lesion appears to be {diagnosis}.",
        "From the description, it is clear that the lesion is {diagnosis}.",
        "The visual and clinical examination suggests the lesion is {diagnosis}.",
        "As per the detailed examination, the lesion was found to be {diagnosis}.",
        "The detailed description indicates that the lesion is {diagnosis}.",
        "Based on the image and clinical details, the lesion was diagnosed as {diagnosis}.",
    ]

    # Collect all non-NaN and non-None iddx values
    diagnoses = [
        data[key]
        for key in ["iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5"]
        if pd.notna(data[key])
    ]

    if diagnoses:
        selected_diagnosis = random.choice(diagnoses)
        template = random.choice(templates)
        return template.format(diagnosis=selected_diagnosis)
    else:
        return None


def generate_report_v2(
    data,
    is_eval=False,
    shuffle=False,
    dropout=0.0,
    include_target=False,
):
    report = []
    patient_info = get_patient_information(data)
    if patient_info:
        report.append(patient_info)
    if not pd.isna(data["clin_size_long_diam_mm"]):
        report.append(
            f"The lesion had a maximum diameter of {round(data['clin_size_long_diam_mm'], 1)} mm."
        )

    if is_eval:
        dropout = 0.0
    if not pd.isna(data["tbp_lv_H"]) and np.random.rand() > dropout:
        report.append(get_hue_info(data))

    if not pd.isna(data["tbp_lv_areaMM2"]) and np.random.rand() > dropout:
        report.append(get_area_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_area_perim_ratio"]) and np.random.rand() > dropout:
        report.append(get_border_jaggedness_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_color_std_mean"]) and np.random.rand() > dropout:
        report.append(get_color_irregularity_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_deltaLBnorm"]) and np.random.rand() > dropout:
        report.append(get_contrast_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_eccentricity"]) and np.random.rand() > dropout:
        report.append(get_eccentricity_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_location_simple"]) and np.random.rand() > dropout:
        report.append(get_location_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_minorAxisMM"]) and np.random.rand() > dropout:
        report.append(get_minor_axis_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_nevi_confidence"]) and np.random.rand() > dropout:
        report.append(get_nevi_confidence_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_norm_border"]) and np.random.rand() > dropout:
        report.append(get_border_irregularity_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_norm_color"]) and np.random.rand() > dropout:
        report.append(get_color_variation_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_perimeterMM"]) and np.random.rand() > dropout:
        report.append(get_perimeter_info(data, is_eval=is_eval))

    if not pd.isna(data["tbp_lv_radial_color_std_max"]) and np.random.rand() > dropout:
        report.append(get_color_asymmetry_info(data, is_eval=is_eval))

    if (
        not pd.isna(data["tbp_lv_symm_2axis_angle"])
        or not pd.isna(data["tbp_lv_symm_2axis"])
        and np.random.rand() > dropout
    ):
        report.append(get_assymetry_info(data, is_eval=is_eval))

    if shuffle and not is_eval:
        if isinstance(shuffle, float):
            if np.random.rand() < shuffle:
                np.random.shuffle(report)
        else:
            np.random.shuffle(report)

    if include_target and "target" in data:
        target_info = get_target_info(data)
        if target_info:
            report.append(target_info)

    return " ".join(report)
