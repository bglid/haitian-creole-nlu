from .context import haitian_creole_nlu


def test_app(capsys, example_fixture):
    # pylint: disable=W0612,W0613
    haitian_creole_nlu.Blueprint.run()
    captured = capsys.readouterr()

    assert "Hello World..." in captured.out
