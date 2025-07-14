import yaml


def test_yaml():
    with open("tests/config.yaml", "rb") as f:
        conf = yaml.safe_load(f.read())

    assert conf["source"]["init_args"]["component"][0]["fields"][0] == "masses"
