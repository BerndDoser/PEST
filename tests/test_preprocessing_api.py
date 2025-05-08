from pest import data_preprocess_api


def test_preprocessing_api():
    """
    Test the data preprocessing function for local images.
    """

    result = data_preprocess_api(
        sim="TNG50-2",
        selection_type="stellar mass",
        min_mass=1e8,
        max_mass=2e8,
        component="stars",
        objects="centrals",
        channels=["Masses"],
        depth=1,
        size=128,
        scale="log",
        orientation="original",
        output_path="./images_test_api/",
        debug=False,
    )

    assert isinstance(result, dict)
    assert "SubID" in result
    assert len(result["SubID"]) == 1


# {'SubID': [79417, 79580, 79811, 83918, 86024],
# 'GroupID': [99, 100, 102, 131, 149],
# 'logMstar': array([10.70847058, 10.710352  , 10.7067737 , 10.70629613, 10.6992251 ]),
# 'logMtot': array([12.27970075, 12.2991249 , 12.42821564, 12.25034364, 12.11282093]),
# 'logMhalo': array([12.22113751, 12.31243984, 12.3679076 , 12.17308883, 12.03844093]),
# 'Rhalf': [2.9622084440507828,
#  5.76645999409507,
#  3.437998228520815,
#  4.131237082964276,
#  5.301299084735755],
# 'SubhaloStarMetallicity': [0.024265184998512268,
#  0.022633448243141174,
#  0.02354022115468979,
#  0.024240750819444656,
#  0.023321228101849556],
# 'SubhaloSFR': [0.0,
#  0.32513657212257385,
#  0.0,
#  0.04936574399471283,
#  4.49891996383667]}
