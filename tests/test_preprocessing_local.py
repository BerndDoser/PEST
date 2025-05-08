import pytest

from pest import data_preprocess_local


@pytest.mark.skip(reason="Skipping test case as per user request")
def test_preprocessing_local():
    """
    Test the data preprocessing function for local images.
    """

    result = data_preprocess_local(
        sim="TNG50-1",
        selection_type="stellar mass",
        min_mass=5e10,
        max_mass=5.2e10,  # [Msun]
        component="stars",
        objects="centrals",
        field="Masses",
        fov="scaled",  # [kpc]
        image_depth=1.0,  #  1 particles per pixel (min. S/N=sqrt(depth))
        image_size=128,
        smoothing=0.0,  # [kpc]
        image_scale="log",
        orientation="original",
        output_path="./images_test_local/",
        debug=False,
    )


# {'SubID': [79417, 79580, 79811, 83918, 86024],
#  'GroupID': [99, 100, 102, 131, 149],
#  'logMstar': array([10.7084704 , 10.71035241, 10.70677389, 10.70629614, 10.69922539]),
#  'logMtot': array([12.27970013, 12.29912528, 12.42821589, 12.25034259, 12.11282079]),
#  'logMhalo': array([12.22113751, 12.31243984, 12.3679076 , 12.17308883, 12.03844093]),
#  'Rhalf': [2.9622568716719884,
#   5.76649192233405,
#   3.4379403007604177,
#   4.131219575001772,
#   5.301296235820272],
#  'SubhaloStarMetallicity': [0.024265185,
#   0.022633448,
#   0.023540221,
#   0.02424075,
#   0.023321228],
#  'SubhaloSFR': [0.0, 0.32513657, 0.0, 0.049365744, 4.49892]}
