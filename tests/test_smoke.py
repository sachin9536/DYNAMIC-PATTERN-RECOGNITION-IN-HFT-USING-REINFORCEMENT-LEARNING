def test_imports():
    import src.utils.logger as logger
    import src.data.loader as loader
    assert callable(logger.get_logger)
    assert callable(loader.load_csv)