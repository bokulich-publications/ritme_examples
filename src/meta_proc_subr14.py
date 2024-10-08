"""Module to process metadata of study by subramanian14"""

import numpy as np
import pandas as pd

# 30.437 is avg. number of days per month
DAYS_PER_MONTH = 30.437


def create_abx_ever(df, host_id="host_id"):
    """Defines abx_ever from abx_7d_prior column"""
    df.sort_values([host_id, "age_days"], inplace=True)

    # find hosts that were exposed to abx at some point
    host_abx = df.loc[df["abx_7d_prior"] == True, host_id].unique()
    host_noabx = df.loc[df["abx_7d_prior"] == False, host_id].unique()

    # set values
    df["abx_ever"] = np.NaN
    df.loc[df[host_id].isin(host_abx), "abx_ever"] = True
    df.loc[df[host_id].isin(host_noabx), "abx_ever"] = False

    return df


def process_table4(path2supp):
    # table 4 with fecal sample information: map sample_id,
    # with host_id and age at sampling
    tab4_df = pd.read_excel(
        path2supp, sheet_name="Supplementary Table 2", header=[1, 2]
    )

    # restructure multiple columns as in https://stackoverflow.com/a/72819472
    tab4_df.columns = pd.MultiIndex.from_tuples(
        [
            (c[1], "") if "Unnamed" in c[0] else (c[0], "") if "Unnamed" in c[1] else c
            for c in tab4_df.columns.to_list()
        ]
    )

    # select only rows with entries - bottom has some notes
    tab4_df = tab4_df.iloc[:996, :].copy(deep=True)

    # rename columns
    tab4_df.rename(
        columns={
            "Cohort": "study_subcohort",
            "Child ID": "host_id",
            "Fecal Sample ID": "sample_id",
            "Age, days": "age_days",
            "Antibiotics within 7 days prior to sample collection": ("abx_7d_prior"),
            # diarrhoea refers to the period within the preceding 7 days and/or at
            # the time of fecal sample collection
            "diarrhoea at the time of sample collection3": ("diag_diarrhea_7d_prior"),
        },
        inplace=True,
    )

    # remove columns
    tab4_df.drop(
        columns=[
            "Age, months",
            "Family ID",
            # if other studies have medication - consider adding
            "Medications (Antibiotics and other) 4",
            "Number of high quality V4-16S rRNA sequences",
            "16S rRNA Sequencing Run ID",
            "Sample specific barcode sequence",
        ],
        level=0,
        inplace=True,
    )

    # 50 healthy bangladeshi children were monitored
    tab4_df["health_status_at_sampling"] = "healthy"
    # mark infants with diarrhea as unhealthy
    tab4_df.loc[
        tab4_df["diag_diarrhea_7d_prior"] == "Yes", "health_status_at_sampling"
    ] = "diarrhea"
    assert len(tab4_df["host_id"].unique()) == 50

    # define diet
    tab4_df["diet_milk"] = np.NaN

    tab4_df.loc[
        np.logical_and(
            tab4_df["Diet at time of fecal sample collection"]["Breast Milk"] == "Yes",
            tab4_df["Diet at time of fecal sample collection"]["Formula1"] == "Yes",
        ),
        "diet_milk",
    ] = "mixed"

    tab4_df.loc[
        np.logical_and(
            tab4_df["Diet at time of fecal sample collection"]["Breast Milk"] == "Yes",
            tab4_df["Diet at time of fecal sample collection"]["Formula1"] == "No",
        ),
        "diet_milk",
    ] = "bd"

    tab4_df.loc[
        np.logical_and(
            tab4_df["Diet at time of fecal sample collection"]["Breast Milk"] == "No",
            tab4_df["Diet at time of fecal sample collection"]["Formula1"] == "Yes",
        ),
        "diet_milk",
    ] = "fd"

    tab4_df.loc[
        np.logical_and(
            tab4_df["Diet at time of fecal sample collection"]["Breast Milk"] == "No",
            tab4_df["Diet at time of fecal sample collection"]["Formula1"] == "No",
        ),
        "diet_milk",
    ] = "no milk"

    tab4_df["diet_weaning"] = np.NaN
    tab4_df.loc[
        tab4_df["Diet at time of fecal sample collection"]["Solid Foods2"] == "Yes",
        "diet_weaning",
    ] = True
    tab4_df.loc[
        tab4_df["Diet at time of fecal sample collection"]["Solid Foods2"] == "No",
        "diet_weaning",
    ] = False
    tab4_df.drop(
        columns=["Diet at time of fecal sample collection"],
        level=0,
        inplace=True,
    )

    tab4_df.columns = tab4_df.columns.droplevel(level=1)

    # strip all string columns from leading and trailing whitespace
    for col in ["study_subcohort", "host_id", "sample_id"]:
        tab4_df[col] = tab4_df[col].str.strip()

    # rename abx values
    tab4_df["abx_7d_prior"] = tab4_df["abx_7d_prior"].replace(
        {"Yes": True, "No": False}
    )
    # create abx_ever feature
    tab4_df = create_abx_ever(tab4_df, "host_id")

    return tab4_df


def process_table1(path2supp):
    # get zygosity information of infants
    tab1_df = pd.read_excel(path2supp, sheet_name="Supplementary Table 1", header=1)
    # again only select rows with info - no notes
    tab1_df = tab1_df.iloc[:50, :].copy(deep=True)

    tab1_df.rename(
        columns={"Child ID": "host_id", "Zygosity": "zygosity"}, inplace=True
    )
    tab1_df = tab1_df[["host_id", "zygosity"]].copy(deep=True)

    # process zygosity entries
    tab1_df["zygosity"] = tab1_df["zygosity"].replace("DZ", "Dizygotic")
    tab1_df["zygosity"] = tab1_df["zygosity"].replace(
        "MZ co-twin in set of triplets", "Monozygotic_triplet"
    )
    tab1_df["zygosity"] = tab1_df["zygosity"].replace(
        "Fraternal co-twin in set of triplets", "Dizygotic_triplet"
    )
    tab1_df["zygosity"] = tab1_df["zygosity"].replace("MZ", "Monozygotic")
    tab1_df["zygosity"] = tab1_df["zygosity"].replace("not tested", "Unknown")
    tab1_df["zygosity"] = tab1_df["zygosity"].replace(np.NaN, "no twins")

    # strip all string columns from leading and trailing whitespace
    tab1_df["host_id"] = tab1_df["host_id"].str.strip()

    return tab1_df


def process_supp_metadata(path2supp):
    # ! table 4: fecal sample information: maps sample_id, with
    # ! host_id and age at sampling
    tab4_df = process_table4(path2supp)

    # ! table 1: get zygosity information of infants
    tab1_df = process_table1(path2supp)

    # ! merge both
    print(f"Shape before merge tab4: {tab4_df.shape}")
    print(f"Shape before merge tab1: {tab1_df.shape}")
    supp_md = tab4_df.merge(tab1_df, on="host_id", how="left")
    print(f"Shape after merge: {supp_md.shape}")

    # note: there are more non-longitudinal infant samples in this cohort
    # after predefined food interventions with SAM (severe accute malnutrition)
    # supp. tab 10 and 11 have more metadata information for these infants
    # maybe include those at some point
    # "Faecal samples were obtained during the acute
    # phase before treatment with Khichuri–Halwa or RUTF, then every 3
    # days during the nutritional rehabilitation phase, and monthly thereafter
    # during the post-intervention follow-up period."

    return supp_md


def _process_sra_metadata(sra_md):
    sra_md.rename(columns={"Sample name [sample]": "sample_id"}, inplace=True)
    # remove columns that we don't need + reformat
    sra_cols_remove = [
        "Organism",
        "Library Source",
        "Library Selection",
        "Bases",
        "Spots",
        "Avg Spot Len",
        "Bytes",
        "Age [sample]",
        "Center Name",
        "Ena-first-public [run]",
        "Ena-last-update [run]",
        "Ena-first-public [sample]",
        "Ena-last-update [sample]",
        "Environment (biome) [sample]",
        "Environment (feature) [sample]",
        "Environment (material) [sample]",
        "External id [sample]",
        "Human gut environmental package [sample]",
        "Host subject id [sample]",
        "Insdc first public [sample]",
        "Insdc last update [sample]",
        "Insdc status [sample]",
        "Investigation type [sample]",
        # 'Library Name',
        "Loader [run]",
        "Name",
        "Project name [sample]",
        "Scientific Name [sample]",
        "Sequencing method [sample]",
        "Submitter id [sample]",
        "Tax ID",
        "Title",
    ]
    sra_md.drop(
        columns=sra_cols_remove,
        inplace=True,
    )
    # rename columns
    sra_md.rename(
        columns={
            "Geographic location (countryand/orsea,region) [sample]": (
                "geo_location_name"
            ),
            "Geographic location (latitude) [sample]": "geo_latitude",
            "Geographic location (longitude) [sample]": "geo_longitude",
            "Sex [sample]": "sex",
            "Target gene [sample]": "exp_target_subfragment",
        },
        inplace=True,
    )
    sra_md["geo_location_name"] = "Bangladesh, Dhaka, Mirpur"

    # add information from reading the study
    # adding experimental information
    # "subjected to polymerase chain reaction (PCR) using primers
    # directed at variable region 4 (V4) of bacterial 16S rRNA genes."
    # from QIITA entry (study ID 1454) - primers taken:
    sra_md["exp_primer"] = "515F [GTGCCAGCMGCCGCGGTAA], 806R [GGACTACHVGGGTWTCTAAT]"
    sra_md["exp_bead_beating"] = True
    sra_md["exp_target_gene"] = "16s rRNA"

    return sra_md


def _postprocess_all_metadata(df_md):
    # ensure correct sorting
    cols = ["host_id", "age_days"]
    df_md = df_md.sort_values(cols)

    # set run ID as index
    df_md.set_index("Run ID", inplace=True)
    df_md.index.name = "id"
    df_md.index = df_md.index.astype(str)

    # replace empty space with _
    df_md.columns = df_md.columns.str.replace(" ", "_")
    # rename
    df_md.rename(
        columns={"Insdc_center_name_[sample]": "insdc_center_name"}, inplace=True
    )
    # make all columns names lowercase
    df_md.columns = df_md.columns.str.lower()

    # transform values to lower case values of specified columns
    ls_lowercase = ["sex", "zygosity"]

    for col in ls_lowercase:
        df_md[col] = df_md[col].str.lower()

    # if zygosity==unknown set to NaN
    df_md.loc[df_md.zygosity == "unknown", "zygosity"] = np.NaN

    # transform strings and digits to boolean values
    df_md = df_md.replace({"diag_diarrhea_7d_prior": {"Yes": True, "No": False}})

    ls_bool = [
        "diag_diarrhea_7d_prior",
        "exp_bead_beating",
        "abx_ever",
        "abx_7d_prior",
        "diag_t1d_at_sampling",
        "diag_seroconv_at_sampling",
    ]

    for col in ls_bool:
        df_md = df_md.replace({col: {1: True, 0: False}})

    # included non-rounded age in months
    df_md["age_months"] = df_md["age_days"] / DAYS_PER_MONTH

    # included 0.5- and fully 1.0 rounded age in months for all studies
    df_md["age_months_rounded05"] = (df_md["age_days"] / DAYS_PER_MONTH * 2).round() / 2
    df_md["age_months_rounded1"] = (df_md["age_days"] / DAYS_PER_MONTH).round()
    return df_md
